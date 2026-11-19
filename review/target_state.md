# `startorch`: target state and the changes to reach it

This document describes the design that `startorch` should have, then the changes to make to the
current implementation (`main`, `b786610`) to get there. The design follows the rules of the
documentation (`docs/docs/index.md`, `docs/docs/sequence.md`, `CLAUDE.md`: modular recipes,
batches of shape `(batch, seq, *)`, reproducibility through an explicit `rng`, `_target_`
configurations, validated arguments, one abstract method to implement a custom generator). Its aim
is a small library that is easy to maintain: **one contract for the shape, one way to pass a
parameter, one class per behaviour, one place for the repr and the configuration**. Breaking changes
are accepted, and the future state is more important than the past one.

Evidence comes from probes and a throw-away prototype run on `main` (torch 2.15, scripts
kept outside the repository; the probes ran in `.venv`, Python 3.14.10). Items marked **[verified]** were reproduced; **[reading]** comes from
the code. Nothing in `src`, `tests` or `docs` was changed. `review/suggestions.md` stays the list of
bugs and chores; this document is the design that makes most of them impossible instead of fixing
them one by one (section 2 says which claims of that file were re-run).

## Constraints

1. **Simple and easy to maintain.** The package is 141 files, 19,881 lines, 225 generator and
   transformer classes, 186 hand-written `__repr__`, 10 copies of `setup_*` / `is_*_config`.
   Distributions alone are 6,014 lines spread over three layers (`random` 1,657, `tensor` 1,986,
   `sequence` 2,371) and unary and binary maths another 3,823 lines (sequence, tensor, tensor
   transformer, dict transformer). The target removes the code that exists only because of the
   design instead of adding code next to it.
2. **One breaking release.** `0.3.0` is an alpha and the README says that the API is not stable.
   No adapter, no alias, no deprecation period: the old names are removed. The release notes list
   the renamings (section 4.3).
3. **Reproducibility is a contract, values may move once.** The order of random draws changes in
   the refactor. The change is made in one release, with golden-value tests that are re-baselined in
   one reviewed PR (section 8).
4. **No slower than today** for the common path (a composite generator on a `(64, 128, 1)` batch).
   Truncated sampling may cost more if it becomes correct (section 6); the gate of section 8 states
   how much.
5. **Optional libraries stay lazy** (`matplotlib`, `plotly`); `torch`, `numpy`, `coola`,
   `batchtensor` and `objectory` are required and are imported at module level (`CLAUDE.md`).
6. **The idea of the library stays**: the user writes the recipe by composing generators, and every
   recipe can be written as a configuration.

## 1. Properties of the package

These are the rules of the documentation. Every design choice below exists to make one of them true
**by construction** (in a base class, a helper or the sampler) instead of by the discipline of each of
the 225 classes.

| Property                    | Statement                                                                                                                                                                                                                                       | Enforced by                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Batch first, caller shape   | A generator fills the `size` it is given: a tensor of shape `(batch, seq, *feature)` for a sequence, `(batch, *)` for an example. All sequences of a batch have the same length. A generator never owns a feature size; the caller does.        | the `generate(size)` contract                  |
| Composable                  | Every generator is a value that can be an argument of another one. Every argument that accepts a generator also accepts a number or a configuration `dict` (`Normal(mean=0.0, std=Uniform(0.5, 1.5))`).                                         | `resolve` in the base class                    |
| Configurable                | Every generator is built from `{"_target_": "startorch.tensor.Normal", ...}`. A configuration of a wrong kind raises `TypeError`. `repr(generator)` lists all the constructor arguments, so it is enough to rebuild the object.                | `Configurable` base class                      |
| Reproducible                | Randomness comes only from the `rng` argument (a `torch.Generator`; `None` means the global torch RNG). The same seed gives the same output. The order of draws is part of the contract: children are drawn in the order of the arguments.      | the base class, golden tests                   |
| Pure                        | `generate` never changes the generator, its arguments or any tensor owned by another generator. The returned tensor belongs to the caller and can be modified in place. The only stateful generator is `Cache`, and it says so.                 | out-of-place ops, copy in `Vanilla` and `Cache` |
| Correct                     | A distribution generator returns samples of the documented law and respects its bounds. A bound is never enforced by a silent `clamp` of a wrong sample.                                                                                       | the sampler, statistical tests                 |
| Strict                      | A wrong argument raises at construction (`ValueError` for a value, `TypeError` for a type) with a message that names the argument and the value. A parameter sampled by a generator is validated with the same message at `generate` time.      | `startorch.utils.validation`                   |
| Device and dtype aware      | The output is on the device of `rng` (CPU if `None`) and has the default floating dtype of torch (`torch.get_default_dtype()`); integer generators return `int64`. Nothing is hard-coded to CPU or `float32`.                                  | the sampler                                    |
| Extensible with one method  | A custom generator is a subclass that implements `generate`. The repr, the configuration and the validation of generator arguments come for free.                                                                                               | `Configurable`                                 |
| Small surface               | One public name per class, one class per behaviour, one naming rule (section 4.2). A behaviour that is a composition of two classes is not a third class.                                                                                       | review rule, `__all__` check                   |
| Lazy optional libraries     | Importing `startorch` does not import `matplotlib` or `plotly`.                                                                                                                                                                                 | `startorch.utils.imports`, `import-linter`     |

Not guaranteed, and documented as such:

- **Identical values across torch versions, devices or dtypes.** The same seed gives the same values
  on the same torch version, device and dtype. Golden tests detect a change; they do not forbid it.
- **Per-example structure.** A *structural* parameter (the period, the AR order, the change point, the
  length of a sequence) is drawn once per `generate` call and shared by the whole batch, because the
  output is one tensor. A *value* parameter is drawn per element. A mixture (`Choice`) is per example.
- **Thread safety of `Cache`.** Every other generator is stateless and can be shared by threads.

## 2. What differs today

| #   | Property                    | Today                                                                                                                                                                                                                                                                                                                                                                                     | Evidence                                        |
| --- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 1   | Caller-owned shape          | Each sequence generator owns a `feature_size` (about 70 signatures). The default is `1` for `RandUniform` and `()` for `RandInt`. A composite takes the shape of its children. Tensor, sequence, time series, example, transition and periodic generators each have a different `generate` signature (6 of them).                                                                         | [reading]                                       |
| 2   | Composable with numbers     | A distribution exists as `X` (parameters are generators), `RandX` (parameters are numbers), `TruncX`, `RandTruncX`, once for tensors and once for sequences: up to 8 classes per distribution. `RandNormalSequenceGenerator` is `TensorSequenceGenerator(RandNormalTensorGenerator())` written again.                                                                                      | [reading], 4,357 lines in the two layers        |
| 3   | Configurable                | `Base.factory(**config)` does **not** check the kind: `A.factory(_target_="...B")` builds a sibling and even `A.factory(_target_="startorch.sequence.RandUniform")` builds a foreign class. `setup_*` then only logs a warning (9 `logger.warning` calls), so a wrong kind fails later with an unrelated `AttributeError`.                                                                  | **[verified]** (`objectory` probe)              |
| 4   | Complete repr               | 186 hand-written `__repr__`. The omissions reported in `suggestions.md` 1.6 for `MulScalar` were **not** reproduced (the sequence version prints `value`); the class of bug is still possible.                                                                                                                                                                                              | [reading], **[verified]** (not reproduced)      |
| 5   | Pure                        | `Add`, `AddScalar`, `Mul`, `MulScalar`, `Fmod` use `add_`, `mul_` on the tensor returned by a child. `Vanilla*` returns views of the stored data. `AddScalar(Vanilla(zeros), 1.0)` called twice leaves the stored data at `2.0`.                                                                                                                                                         | **[verified]**                                  |
| 6   | Correct: tails              | Truncated laws are `icdf(uniform(cdf(min), cdf(max))).clamp(min, max)`. For bounds in the far tail `cdf(min) == cdf(max)`, `icdf` returns `inf` and the clamp hides it: `rand_trunc_normal((5,), 0, 1, 9, 10)` returns `[10, 10, 10, 10, 10]`.                                                                                                                                              | **[verified]**                                  |
| 7   | Correct: arguments ignored  | `make_sparse_uncorrelated_regression` validates `noise_std` and never uses it (`std=torch.ones(...)`): the residual std is 1.007 for `noise_std=0` and for `100`. `make_markov_chain` with a 1-D `init` gives the same first state to every row (`[2, 2, 2, 2, 2, 2, 2, 2]`). `make_linear_regression` adds the noise to the features, not to the target, unlike scikit-learn and Friedman. | **[verified]** (first two), [reading] (third)   |
| 8   | Per-example mixture         | `MultinomialChoice*` draws one child for the whole batch: three rows of `Choice(Full(0), Full(1))` are all `1.0`.                                                                                                                                                                                                                                                                       | **[verified]**                                  |
| 9   | Strict                      | 22 `RuntimeError`, 119 `ValueError`, 3 `TypeError` for argument errors; `check_integer_ge(True, ...)` accepts a `bool`; `check_std` says "greater than 0" and accepts 0; `period == 0` raises a bare `ZeroDivisionError` at generate time.                                                                                                                                              | **[verified]** (bool), [reading] (the others)   |
| 10  | Device and dtype            | `torch.randn(size, generator=rng)` and friends are called without `device`; a CUDA generator cannot be used. Everything is `float32`.                                                                                                                                                                                                                                                  | [reading]                                       |
| 11  | Small surface               | Every class is exported twice (`Abs` and `AbsSequenceGenerator`), and not consistently: `startorch.sequence` has no `Vanilla`. The keyword for the generator is `rng` in `generate` and `generator` in `startorch.random` and in `make_*`. `ConcatenateTimeSeriesGenerator` does not concatenate; it merges keys and a duplicated key silently overwrites. `constants.SEQUENCE`, `SEQ_LEN`, `BATCH_SIZE` are unused. | **[verified]** (`ImportError`), [reading]       |
| 12  | Duplicated families         | Elementwise maths exist four times (tensor generator, sequence generator, tensor transformer, dict transformer). `Vanilla`, `Cache`, `Choice`, `Transform`, `Concatenate` exist at several levels. `periodic/` (616 lines) re-implements `ceil(seq_len / period)`, `repeat`, `slice` in four places. `transition/` is a tensor generator of size `(n, n)` with three small extras. | [reading]                                       |
| 13  | Dead or unnamed code        | `Mixed` / `mix2sequences` (swaps the odd positions of two keys) is used nowhere in the docs or the notebooks and is mutating. `utils/imports.py` has 238 lines for three packages, including three `objectory` helpers for a required dependency.                                                                                                                                       | [reading], `grep` of docs and notebooks         |

## 3. Architecture

### 3.1 The contract: caller-owned shape

There are two kinds of generator, distinguished by what they return, plus transformers.

| Kind               | Call                                  | Returns                                          | Package                  |
| ------------------ | ------------------------------------- | ------------------------------------------------ | ------------------------ |
| `TensorGenerator`  | `generate(size, *, rng=None)`         | a tensor of shape `size`                         | `startorch.tensor`       |
| `ExampleGenerator` | `generate(size, *, rng=None)`         | a `dict[str, Tensor]`; every value starts with `size` | `startorch.example`      |
| `TensorTransformer`| `transform(tensor, *, rng=None)`      | a tensor                                         | `startorch.transformer`  |
| `ExampleTransformer` | `transform(data, *, rng=None)`      | a `dict[str, Tensor]`                            | `startorch.transformer`  |

- **A sequence generator is a tensor generator called with `size=(batch, seq, *feature)`.** There is
  no `BaseSequenceGenerator` and no `feature_size` argument. A generator that needs a time axis
  (`Arange`, `Cumsum`, `AutoRegressive`, `Markov`, `Wiener`, `Periodic`, `Cat`, `Sort`, `Time`, ...)
  lives in `startorch.sequence`, subclasses `TensorGenerator` and calls
  `check_sequence_size(size)` (at least 2 dimensions, `seq > 0` where it matters). A generator that is
  elementwise (`Normal`, `Add`, `Abs`, ...) lives in `startorch.tensor` and is a sequence generator
  for free. This deletes the sequence copy of every distribution and every elementwise operation.
- **A time series is an example whose `size` is `(batch, seq)`**; a tabular example has
  `size=(batch,)`. `startorch.timeseries` disappears as a kind; its two real classes move to
  `startorch.example` (section 3.7).
- **`size` is a tuple of ints, keyword-only `rng`.** The call
  `Normal().generate((4, 128, 1))` replaces `RandNormal().generate(seq_len=128, batch_size=4)`.
  The cost is every example of the README, the user guide and the notebooks; the gain is that the
  feature shape is explicit and uniform (decision D1).
- **`rng` is the only name.** `startorch.random` and the `make_*` functions use `rng` too.

### 3.2 `Configurable`: repr, configuration and resolution once

One small base class (about 80 lines) replaces 186 `__repr__`, 10 `setup_*`, 10 `is_*_config`:

```python
class Configurable(ABC, metaclass=AbstractFactory):
    def __repr__(self) -> str: ...        # from inspect.signature(type(self).__init__)
    @classmethod
    def resolve(cls, value): ...          # instance -> itself, dict -> cls.factory(**value),
                                          # else TypeError; checks isinstance(result, cls)
```

- **Repr by convention.** The constructor argument `x` is stored as `self._x` (this is already the
  convention of every class). The repr prints each constructor argument with `repr()`; a nested
  generator is indented the way `coola.formatting` does today. A test checks, for **every** concrete
  class found in the registry, that `repr(obj)` mentions each constructor argument. A class cannot
  omit a field.
- **`resolve` checks the kind.** Each base class (`TensorGenerator`, `ExampleGenerator`,
  `TensorTransformer`, `ExampleTransformer`) is a factory root. `resolve` instantiates a `dict`
  through the factory and then verifies `isinstance`; a wrong kind raises
  `TypeError("expected a TensorGenerator, got SequenceFoo")`. The `logger.warning` and the noisy
  `info` log are deleted.
- **`TensorGenerator.resolve` also accepts a number** and returns `Full(value)`. This is what makes
  `Normal(mean=0.0, std=Uniform(0.5, 1.5))` and `Add([signal, 3.0])` work (section 3.3). A `bool` is
  rejected.
- The `is_*_config` functions are replaced by one `is_config(config, base)` in `startorch.utils`
  (it must not import another `startorch` package).
- `AbstractFactory` stays as the registry. Whether `coola.factory` can replace `objectory` is a
  separate decision (D8): `coola 2` is **not installed** here (`uv.lock` has `coola 0.11.1`), so the
  claims of `suggestions.md` section 7.1 about it were not verified.

### 3.3 One class per distribution, parameters are values or generators

Today a parameter is either a float (`RandX`) or a generator (`X`). The target has one class where
every parameter accepts a number, a configuration or a generator:

```python
Normal(mean=0.0, std=1.0)                                  # was RandNormal
Normal(mean=Uniform(-1.0, 1.0), std=Uniform(0.5, 1.5))     # was Normal(RandUniform, RandUniform)
Normal(mean=0.0, std=1.0, min_value=-2.0, max_value=2.0)   # was RandTruncNormal
```

- **Resolution at construction.** `Full(c)` is a constant generator. It draws no random number, so a
  recipe that uses only numbers draws exactly what `RandX` draws today. **[verified]** by the
  prototype (section 6): `Normal(0.0, 1.0)` costs the same as `RandNormal` (88 us vs 86 us on a
  `(64, 128, 1)` batch); the extra `torch.full` of the constants is not measurable.
- **Distributions after the change** (tensor level only): `Uniform`, `LogUniform`, `AsinhUniform`,
  `Normal`, `HalfNormal`, `LogNormal`, `Cauchy`, `HalfCauchy`, `Exponential`, `Poisson`, `Integer`
  (was `RandInt`), `Multinomial`. The 6 continuous laws with an unbounded support take the optional
  `min_value` / `max_value` that replace the `Trunc*` classes.
- **Validation.** Numbers are validated at construction (`std > 0`, `min_value <= max_value`,
  `low > 0` for `LogUniform`, `rate > 0`). A sampled parameter is validated at `generate` with the
  same message (`torch.any(std <= 0)`). A dataset `noise_std` is `>= 0` (0 means no noise); a
  distribution scale is `> 0`. The messages are fixed accordingly (the "greater than 0" that
  accepts 0 disappears).
- **Shape.** Parameters are drawn with the full `size` (one value per element), as today. A
  per-sequence parameter is `Constant(...)` (section 3.5), as the user guide shows.

### 3.4 `startorch.random`: one correct sampler

`random/` keeps stateless functions (`normal`, `uniform`, ... with an `rng` keyword) that the
generator classes call. The 1,038-line `bounded.py` (about 60 repeated `torch.any(x.le(0))` blocks
and 18 near-identical functions) becomes a small table:

- Each continuous law is described by `cdf`, `icdf`, `sf` (survival) and `isf` in **float64**,
  written with the special functions of torch.
- One function `sample_truncated(law, loc, scale, min_value, max_value, size, rng)` draws
  `u ~ U(0, 1)` and inverts on the side where the mass is: with the survival function when the lower
  bound is in the upper half of the law, with the cdf otherwise. There is **no clamp that hides a
  wrong sample**; the only clamp absorbs rounding to the bound and a test asserts that it never moves
  a value by more than `1e-6` of the interval.
- The unbounded path stays `randn * std + mean` (no inversion, no float64).
- **Pitfall found by the prototype [verified]:** `torch.special.ndtr(-9.0)` returns `0.0` even in
  float64 (the true value is `1.1e-19`), whereas `torch.special.log_ndtr` is accurate. The sampler
  must use `exp(log_ndtr(.))`; a test pins `ndtr`-free tails (`min_value=9, max_value=10`).
- A shared `check_*` set in `startorch.utils.validation` replaces the inline checks
  (`check_positive`, `check_ordered`, `check_same_shape`, ...); `check_integer_ge`, `check_interval`
  reject `bool`.
- Today every law has a `rand_x` function (scalar parameters) and an `x` function (tensor
  parameters), except Poisson, which has only `rand_poisson`. The target keeps **one** function per
  law, `x(..., rng)`, with tensor parameters: the classes resolve numbers into constants, so the
  scalar `rand_*` variants (and their copies of the validation) are deleted. A `poisson(rate, rng)`
  function completes the set.

### 3.5 Elementwise operations: written once, as transformers and combinators

The 3,823 lines of maths collapse to the following (all tensor level; they work on sequences):

| Need                                        | Target                                                                                                                       | Replaces                                                                  |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Apply a function to the output of a generator | `Transform(generator, transformer)` (tensor level) and `Transform(generator, transformer)` (example level)                  | `Abs*Generator`, `Exp*Generator`, ... (about 40 wrapper classes)          |
| Parameter-free functions (`abs`, `ceil`, `exp`, `expm1`, `floor`, `frac`, `log`, `log1p`, `neg`, `rsqrt`, `sigmoid`, `sqrt`, `acosh`, `asinh`, `atanh`, `cosh`, `sinh`, `tanh`, `sinc`, ...) | one transformer `Unary(op)` where `op` is a key of a table `{"abs": torch.abs, ...}`; an unknown name raises `ValueError` listing the keys | about 25 `*TensorTransformer` classes, the dict versions                  |
| Functions with arguments                    | `Clamp(min, max)`, `Pow(exponent)`, `Round(decimals)`, `Logit(eps)`, `Cast(dtype)`, `Normalize(p, dim)`                       | `Float*`, `Long*`, `Clamp*`, `Pow*`, `Round*`, `Logit*`, `Normalize*`     |
| Combine generators                          | `Add(terms)`, `Mul(factors)` (n-ary), `Sub(left, right)`, `Div(left, right, rounding_mode=None)`, `Fmod(left, right)`; any operand may be a number | `Add*`, `AddScalar*`, `Mul*`, `MulScalar*`, `Div*`, `Fmod*`, `Neg*`        |
| Combine keys of an example                  | `Binary(op, left, right, output)` over a table of five ops                                                                    | `AddTransformer`, `SubTransformer`, `MulTransformer`, `DivTransformer`, `FmodTransformer` |
| Apply a tensor transformer to a key         | `Apply(transformer, input, output, exist_ok=False)`                                                                           | `TensorTransformer`, `BaseTensorTransformer` (dict)                        |

- **All operations are out of place.** `Add([a, b])` is `a + b`; the result is a new tensor, so
  property *Pure* holds without a convention (the `1.3` bug of `suggestions.md` cannot occur) and
  an integer generator plus `0.5` promotes correctly instead of raising a dtype error.
- **`Vanilla` and `Cache` return a copy** (`clone()`), the price of owning the output. `Cache` keeps its
  own tensor and hands out copies; `deepcopy` disappears as an option.
- The tensor transformer `Poisson` / `Exponential` (noise from a sampled rate) are kept because they
  use `rng`; they are two classes.

### 3.6 Sequence generators (need a time axis)

`startorch.sequence` keeps only generators whose meaning depends on dimension 1:

| Class                                             | Notes for the target                                                                                                                                                                   |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Arange`, `Linear`, `SineWave` (`value`, `frequency`, `phase`, `amplitude`), `Wiener` | `Wiener` takes the caller's feature shape (today it hard-codes one feature). `Linear` and `SineWave` stay elementwise on their child generators.                |
| `Constant(generator)`                             | draws `(batch, 1, *feature)` and expands over time. Returns a copy.                                                                                                                    |
| `Cumsum`, `Sort`, `Time` helpers                  | `Time` factories (`create_*_time_diff`, 8 classmethods) become module functions that return a generator; the class is not needed.                                                       |
| `Periodic(generator, period)`                     | `period` is an int or a generator; draws it once per call, generates `(batch, period, *feature)` and tiles to `seq`. Validates `period >= 1`. Replaces `PeriodicSequence`, `RepeatPeriodic*`, `PeriodicTimeSeries` and the package `startorch.periodic`. |
| `Cat(generators, changepoints)`                   | N generators and N-1 change points (was `Cat2`). Change points are structural: one draw per call.                                                                                    |
| `AutoRegressive(value, coefficient, noise, order, max_abs_value, warmup)` | Wraps with `clamp` instead of `fmod` (a modulo creates jumps that are not an AR process). The Python loop stays and is documented as the slowest generator. `order >= 1` is checked in `__init__` for a constant and at generate time otherwise (`ValueError`). |
| `Markov(transition, init)`                        | `transition` is a `TensorGenerator` of size `(n, n)`; `init` is `(n,)` or `(batch, n)` (a 1-D `init` is expanded: the verified bug disappears). Shapes are validated.                   |
| `Vanilla(data)`                                   | slices the stored tensor, returns a copy. A `size` larger than the data raises `ValueError`.                                                                                           |

- `startorch.transition` disappears as a kind. `Diagonal` becomes the tensor generator `Eye`,
  `PermuteDiagonal` becomes `Permutation`, `Normalize` becomes the transformer `Normalize(p=1)`
  (the default `p=2` produced rows that are not probabilities), and a transition matrix is
  `Transform(Uniform(), Normalize(p=1))` called with `size=(n, n)`.
- **Removed:** `Mixed` / `mix2sequences` (unused, mutating, not self-describing). A random-period sine
  with an integer number of cycles per period (`SineWavePeriodic`) is **not** reproduced by
  `Periodic(SineWave(...))` when the period is sampled; it is a functionality reduction to confirm
  (decision D5).

### 3.7 Example generators (dicts of tensors)

| Class                                 | Target                                                                                                                                                                                        |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Fields(generators, feature_size=())` | One class for `TensorExampleGenerator` and `SequenceTimeSeriesGenerator`: each value is `generator.generate((*size, *feature_size))`. `feature_size` may be a mapping key -> shape.            |
| `Union(generators)`                   | Replaces `ConcatenateExample` and `ConcatenateTimeSeries`: merges the keys of several generators and **raises `KeyError` on a duplicated key** (today it silently overwrites).                   |
| `Merge(generators, time_key)`         | Merges time series along the sequence axis sorted by time (was `MergeTimeSeriesGenerator`).                                                                                                   |
| `Choice(generators)`                  | Per example (section 3.8), for dicts.                                                                                                                                                         |
| `Transform(generator, transformer)`   | applies an `ExampleTransformer`.                                                                                                                                                              |
| `Cache`, `Vanilla`                    | stateful / data-backed, return copies.                                                                                                                                                        |
| `VariableLength(generator, seq_len)`  | draws the sequence length of a time series per call (was `TimeSeriesExampleGenerator`; the plural `generators` argument that took one generator is fixed).                                    |
| `Periodic(generator, period)`         | same rule as section 3.6, for a dict.                                                                                                                                                         |

- **Datasets** (`BlobsClassification`, `CirclesClassification`, `MoonsClassification`,
  `HypercubeClassification`, `LinearRegression`, `Friedman1/2/3`, `SparseUncorrelated`, `SwissRoll`)
  stay as `make_*(num_examples, ...)` functions with a class wrapper that takes the arguments of the
  function. Targets for each:
  - all accept `rng` (not `generator`) and validate with `startorch.utils.validation`;
  - `make_sparse_uncorrelated_regression` uses `noise_std` (the default stays `0.0`, so the
    behaviour changes: the targets become noise free by default; a regression test fails without the fix);
  - `make_linear_regression` adds the noise to the **target**; a `feature_noise_std` argument is
    added only if a maintainer wants errors-in-variables (D6);
  - `make_friedman3_regression` guards the denominator (`rand_uniform` can return exactly 0);
  - `HypercubeClassification` is documented as "one-hot plus noise" (the name is kept, D6);
  - the magic seeds (`13683624337160779813`, `17532042831661189422`) become `DEFAULT_SEED` in
    `startorch.constants`.
- The class wrappers drop the boilerplate properties that duplicate the constructor arguments
  (`feature_size`, `noise_std`, ...): the attribute `_x` is private, the repr shows the value.

### 3.8 Mixtures are per example

`Choice(generators)` (tensor and example level) draws one index **per example** (`dim 0`) and
generates each child only for the rows it was chosen for (`child.generate((n_i, *size[1:]))`), then
scatters the rows into the output. Cost is proportional to the output, not to the number of
children. The weights are validated by one helper (`prepare_weighted_generators` stays in
`startorch.utils`, with `nan` / `inf` / empty checks added). Batch-level choice, which is what
`MultinomialChoice*` does today, is a documented non-goal: a user who wants it draws the index
outside.

### 3.9 Plots

`plot/matplotlib` and `plot/plotly` (461 lines, mirror images) stay, with a shared data-preparation
function in `startorch.plot.utils` (no matplotlib import) used by both backends:

- `plot_sequence` draws **one line per feature** (today `.flatten()` interleaves the features into
  one zig-zag), with the same arguments in both backends (`xscale`, `yscale`, `figsize`).
- `to_array` uses `.detach().cpu().numpy()`.
- Both modules keep the lazy-import guard; the `Mock()` fallbacks are deleted if the guard is enough.

### 3.10 Utilities, constants and packaging

- `startorch.utils` keeps `validation` (extended, section 3.4), `seed` (`get_torch_generator`,
  `setup_torch_generator`; `get_random_seed` only if used), `weight`, `mask` (vectorised:
  `argsort` of one `rand` call instead of a loop over rows), `conversion` (`to_tuple`, `to_tensor`),
  `format` (`str_target_object` moves with `Configurable`), `imports` (two families, `matplotlib` and
  `plotly`, generated by one small factory function; the three `objectory` helpers are deleted),
  `tensor.circulant` (with an empty-input check).
- `constants.py` keeps `TARGET`, `FEATURE`, `TIME`, `VALUE`, `DEFAULT_SEED`; `SEQUENCE`, `SEQ_LEN`,
  `BATCH_SIZE` are deleted.
- Add `src/startorch/py.typed` (the classifier `Typing :: Typed` is already declared) and
  `startorch.__version__`.
- A shared alias `Size = tuple[int, ...]` (in `startorch.utils.typing`) replaces the mixture of
  `tuple[int, ...] | list[int] | int` in about 70 signatures.

## 4. Public API after the change

### 4.1 Layout

```text
startorch/
  base.py          Configurable, is_config                          (~120 lines)
  constants.py     TARGET, FEATURE, TIME, VALUE, DEFAULT_SEED
  random/          law table, sample_truncated, discrete samplers   (~450)
  tensor/          TensorGenerator + elementwise generators:
                   distributions (12), Full, Eye, Permutation,
                   Add, Sub, Mul, Div, Fmod, Transform, Choice,
                   Shaped (adds a feature shape)                    (~900)
  sequence/        Arange, Constant, Cumsum, Sort, Time, Linear,
                   SineWave, Wiener, AutoRegressive, Markov, Cat,
                   Periodic, Vanilla                                (~900)
  example/         ExampleGenerator, Fields, Union, Merge, Choice,
                   Transform, Cache, Vanilla, VariableLength,
                   Periodic, datasets (9)                           (~1,100)
  transformer/     TensorTransformer: Unary, Clamp, Pow, Round,
                   Logit, Cast, Normalize, Identity, Sequential,
                   Poisson, Exponential
                   ExampleTransformer: Apply, Binary, Select,
                   Remove, Lut, Linear, Wave, Identity, Sequential  (~600)
  plot/            matplotlib, plotly, shared preparation           (~350)
  utils/           validation, seed, weight, mask, conversion,
                   imports, typing                                  (~450)
```

The line counts are **estimates** (not measured); they total about 4,900 lines against 19,881
today, and about 80 classes against 225. The tests shrink in the same proportion (21,236 lines
today), because the duplicated families disappear with their tests.

### 4.2 Naming rule

| Concept                     | Rule                                                                                                      |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| Name of a class             | One public name, no `Generator` suffix inside a kind-specific package (`startorch.tensor.Normal`).        |
| Number or generator         | The same class. No `Rand` prefix.                                                                         |
| Truncated law               | The same class with `min_value` / `max_value`. No `Trunc` prefix.                                         |
| Needs a time axis           | The class is in `startorch.sequence`. Otherwise it is in `startorch.tensor`.                              |
| Returns a dict              | The class is in `startorch.example`.                                                                      |
| Factory classmethod         | `create_*`, everywhere.                                                                                   |
| Argument of a function      | `rng` for a generator of randomness; options after the first required argument are keyword-only.        |

### 4.3 Old names to new names (release notes)

| Old                                                                                      | New                                                          |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `sequence.RandX`, `sequence.X`, `sequence.RandTruncX`, `sequence.TruncX`, `tensor.*` same | `tensor.X(..., min_value, max_value)`                        |
| `sequence.RandInt`                                                                       | `tensor.Integer`                                             |
| `sequence.Abs`, `Exp`, ... , `tensor.Abs`, ..., `tensor.transformer.AbsTensorTransformer` | `transformer.Unary("abs")`, used with `Transform`            |
| `sequence.Add`, `AddScalar`, `Mul`, `MulScalar`, `Div`, `Fmod`, `Neg`                     | `tensor.Add`, `Mul`, `Sub`, `Div`, `Fmod` (numbers as operands) |
| `sequence.Full`, `tensor.Full`                                                           | `tensor.Full`                                                |
| `sequence.Tensor` (`TensorSequenceGenerator`)                                            | `tensor.Shaped`                                              |
| `sequence.MultinomialChoice`, `timeseries.MultinomialChoice`, `tensor.MultinomialChoice` | `tensor.Choice`, `example.Choice` (per example)              |
| `sequence.Cat2`                                                                          | `sequence.Cat`                                               |
| `sequence.Periodic`, `periodic.*`, `timeseries.Periodic`                                 | `sequence.Periodic`, `example.Periodic`                      |
| `transition.*`                                                                           | `tensor.Eye`, `tensor.Permutation`, `transformer.Normalize`  |
| `timeseries.Sequence`, `example.Tensor`                                                  | `example.Fields`                                             |
| `timeseries.Concatenate`, `example.Concatenate`                                          | `example.Union` (duplicate key raises)                       |
| `timeseries.Mixed`                                                                       | removed                                                      |
| `setup_*`, `is_*_config`                                                                 | `Base.resolve(value)`, `is_config(config, Base)`             |
| `generate(seq_len, batch_size, rng)`                                                     | `generate(size, *, rng=None)`                                |
| `generator=` in `startorch.random` and `make_*`                                          | `rng=`                                                       |

## 5. Dependents and dependencies

- **Inside the repository.** `docs/docs/sequence.md`, `docs/docs/index.md`, the README and the four
  tracked notebooks use `generate(seq_len=..., batch_size=...)`, `RandX`, `Add`, `Constant` and
  `setup_*`. They are rewritten in the same release; the notebooks are executed in CI afterwards
  (section 8). `mkdocs.yml` loses the pages `periodic`, `timeseries`, `transition` and
  `tensor_transformer`, gains `transformer` and a user guide (section 7, step 10).
- **Import contracts.** `startorch.utils` still imports no other `startorch` package;
  `startorch.base` imports only `startorch.utils`. The existing `import-linter` contracts are updated
  for the removed packages.
- **Dependencies stay the same.** `batchtensor` is kept (23 import lines of nested and per-axis
  helpers, used by the sequence and dict generators); `objectory` is kept (D8). Nothing in the target needs `coola 2`.
- **External users.** The API is documented as unstable until 1.0; no compatibility layer.

## 6. Evidence: a prototype of the parametric, truncated generator

A throw-away prototype (`Configurable`-style repr, `param()` resolution, `Normal` with optional
bounds, float64 survival-function sampler; about 90 lines, kept outside the repository) was run
against the current `RandNormal` and `RandTruncNormal` (torch 2.15, CPU, `size=(64, 128, 1)`).

| Check                                                         | Today                       | Prototype                                         |
| ------------------------------------------------------------- | --------------------------- | ------------------------------------------------- |
| Truncated law in the far tail, `min_value=9, max_value=10`    | `[10, 10, 10, 10, 10]`      | 81,836 distinct values in `[9.0, 9.98]` for 100,000 draws |
| KS statistic, `N(0,1)` truncated to `[0.5, 1.5]` (100,000)    | not tested                  | 0.0025 (5% critical value 0.0043)                 |
| KS statistic, law conditional on `x >= 9` (50,000)            | not testable                | 0.0028 (5% critical value 0.0061)                 |
| Unbounded `Normal(0, 1)` with constants resolved to `Full`    | 86-89 us / call             | 88-102 us / call (same within noise)              |
| Truncated to `[-1, 1]`                                        | 182-191 us / call           | 950-1,210 us / call (**about 5x slower**)         |
| Same seed twice                                               | equal                       | equal                                             |
| `Normal("a")`                                                 | `ValueError: could not convert string to float` | `TypeError: expected a number or a generator` |
| `repr(Normal(0.0, Normal(1.0, 0.1, min_value=0.5)))`          | hand-written                | lists every argument, nested                      |

Findings that change the design:

1. **`torch.special.ndtr` is not usable in the tail**: `ndtr(-9.0)` is `0.0` in float64. The first
   version of the prototype used it and reproduced the collapse (`[10, ...]`, KS `nan`). With
   `exp(log_ndtr(.))` the law is exact. This is why section 3.4 forbids `ndtr` and tests the tail.
2. **The correct truncated sampler is slower** (float64, both branches of the survival and the cdf are
   computed). 1 ms for 8,192 values is acceptable for a data generator, but the gate of section 8
   states the budget, and an optimisation is allowed (compute only the branch needed when the bounds
   are scalars, use float32 when both bounds are within 4 standard deviations of the mean).
3. **Resolving numbers to `Full` is free** and keeps the draws of `RandX`, so the Cartesian product of
   classes can be deleted without a speed cost on the common path.

The prototype did not cover `Choice`, the example generators and the sequence generators. Their
design (sections 3.6 to 3.8) comes from reading the code and should be prototyped in step 5 of
section 7 before the old classes are deleted.

## 7. Changes to reach the target state

One branch and one pull request per step, stacked where a step depends on the previous one
(`CLAUDE.md`). Steps 1 to 4 are independent of the shape contract and can be merged on their own.

| Step | Change                                                                                                                                                                                                                                                  | Result                                                                  |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 1    | **Bug-fix PRs**, each with a regression test that fails without the fix: Markov `init`, `noise_std` of sparse-uncorrelated, `period == 0`, `LogUniform` with `low <= 0`, `bool` as `int`, `check_std` message. Delete the dead constants and the commented code. | Real bugs fixed whatever the redesign.                                  |
| 2    | **`Configurable`** (repr, `resolve`, `is_config`) with the repr-completeness test, applied to the existing classes by deleting their `__repr__` and `setup_*`. `TypeError` for a wrong kind.                                                         | Section 3.2; about 180 `__repr__` and 10 `setup_*` removed; no behaviour change except the repr format and the `TypeError`. |
| 3    | **Sampler.** `random/` table of laws, `sample_truncated` with the survival function and float64, `rng` keyword, device from `rng`. Statistical tests (KS against the analytic cdf) and the far-tail test.                                              | Section 3.4; bug 6 fixed; `bounded.py` shrinks.                          |
| 4    | **Out-of-place combinators and copies** in `Vanilla` and `Cache`; the aliasing tests (`generate` twice, mutate the output, `Vanilla` data unchanged).                                                                                                  | Property *Pure*; bug 5 fixed.                                           |
| 5    | **Shape contract and tensor level.** `TensorGenerator.generate(size, *, rng)`, one class per distribution with number-or-generator parameters and `min_value` / `max_value`; `Full`, `Add`, `Sub`, `Mul`, `Div`, `Fmod`, `Transform`, `Choice` (per example), `Shaped`. Golden-value tests of the new classes. | Sections 3.1, 3.3, 3.5, 3.8; the `tensor` package is final.             |
| 6    | **Sequence level.** Move the time-axis generators to `TensorGenerator` with `check_sequence_size`; delete the sequence copy of the distributions and the maths; `Periodic`, `Cat`, `Markov` on tensor generators; delete `startorch.periodic` and `startorch.transition`. | Section 3.6; about 3,800 lines deleted.                                 |
| 7    | **Transformers.** `Unary`, `Clamp`, `Pow`, `Round`, `Logit`, `Cast`, `Normalize`, `Apply`, `Binary`; delete the 40 duplicated classes.                                                                                                                 | Section 3.5.                                                            |
| 8    | **Examples.** `ExampleGenerator.generate(size)`, `Fields`, `Union` (raises on a duplicate key), `Merge`, `VariableLength`, `Periodic`; datasets use `rng`, validate with `validation`, fix the noise of `make_linear_regression`; delete `startorch.timeseries`. | Section 3.7.                                                            |
| 9    | **Plots** with shared preparation and one line per feature; `py.typed`, `__version__`, `Size` alias, `imports.py` factory, `DEFAULT_SEED`.                                                                                                             | Sections 3.9, 3.10.                                                     |
| 10   | **Docs.** Rewrite the README, `index.md`, `sequence.md`; add the user guide pages listed in section 8; generate the list of generators from `__all__`; execute the notebooks in CI; add `CHANGELOG.md` with the renaming table of section 4.3.        | Documentation matches the code, with the PR numbers.                    |

Steps 5 to 8 are the breaking release. They can be delivered as one stacked series merged together,
so that `main` never contains two APIs.

## 8. Tests and gates

The tests keep the rules of `CLAUDE.md` (one behaviour per test, `match` on every `raises`, seeded
data, 100% coverage, patched tests with an integration counterpart). New or changed:

- **Statistical tests** (`tests/integration/test_distributions.py`): for each law, `n = 10,000`, fixed
  seed, KS statistic against the analytic cdf, with and without bounds, including the far tail
  `[9, 10]` and the lower tail. A bound is never violated and no value sits exactly on a bound
  unless the interval is degenerate. (A KS test needs `scipy`, which is **not installed** here: use
  the closed-form cdf of `torch.distributions`, as the prototype did, or add `scipy` to the dev group.)
- **Golden values**: for a handful of recipes per family, `seed -> output` stored in the test
  (`tests/unit/golden/`). Changes are reviewed as a diff, not discovered.
- **Smoke test of every class**: a parametrised test builds each registered class from its
  `_target_` configuration with minimal arguments, calls `generate((2, 5, 1))` or the example
  equivalent, and checks the shape, dtype, finiteness, and `repr`. It replaces most of the
  hand-written shape tests, and runs without patching (the integration counterpart of the patched
  unit tests).
- **Repr completeness and round trip**: for every concrete class, `repr` mentions every constructor
  argument; the objects built from the config of `repr`-visible arguments are equal.
- **Contracts**: `generate` called twice with the same seed is equal; with `rng=None` it uses the
  global RNG; the generator and its arguments are unchanged after `generate`; the output can be
  modified in place without changing the next call; a wrong kind in a configuration raises
  `TypeError`; a CUDA `rng` gives CUDA tensors (marked `cuda`, skipped without a GPU).
- **Edge cases** for composites: `seq = 0`, `batch = 0`, `size` with one dimension for a sequence
  generator (raises `ValueError`), a `Choice` with a single child or an unlucky draw that gives a
  child no row.
- **Benchmark gate** (`benchmarks/`): the unbounded `Normal`, `Add` of three sine waves and
  `AutoRegressive` on `(64, 128, 1)` must not be slower than `main` by more than 10%; the truncated
  `Normal` must stay under 5x of `main` (the prototype measured about 5x; an optimisation of
  section 6 is expected). The benchmark also reports `Choice` on `(4096, 16)`.
- **CI**: run the integration tests and the doctests with `--cov-fail-under=100`; execute the
  notebooks (`nbmake`) in the nightly job; `ruff`, `ty` and `import-linter` as today.
- **Documentation tests**: the user guide pages are doctests; each page has a "recipe" that is also a
  configuration, tested for equality with the object built in Python.

User guide pages to add (the guide has one page, `sequence.md`, and the `quickstart.md` entry is
commented out): *Quickstart*, *Shapes and the `size` contract*, *Composing generators and
configurations*, *Reproducibility and the RNG contract* (including the order of draws), *Writing a
generator*, *Examples and time series*, *Naming rules* (section 4.2).

## 9. Decisions for the maintainers

Each decision has a recommendation; the design above assumes the recommendation.

| #   | Decision                                                                                       | Recommendation                                                                                                                                                              | Cost of the alternative                                                                                                  |
| --- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| D1  | Replace `generate(seq_len, batch_size)` by `generate(size)` and remove `feature_size`          | **Yes.** One contract is what lets the sequence copies of every elementwise class be deleted.                                                                                | Keeping a `BaseSequenceGenerator` with `feature_size` keeps about 3,800 lines and the two-layer product of section 2.    |
| D2  | Merge `X` / `RandX` / `TruncX` / `RandTruncX` into one class with number-or-generator parameters | **Yes.**                                                                                                                                                                    | Keeping `RandX` as thin subclasses doubles the public names and the docs.                                                |
| D3  | `Choice` per example                                                                           | **Yes.** It is what a mixture means, and the cost is proportional to the output.                                                                                              | Batch-level choice is a one-line wrapper if someone needs it.                                                            |
| D4  | Truncated sampling in float64 with the survival function, about 5x slower                      | **Yes**, with the optimisation allowed by the gate. A correct sampler is more valuable than a fast wrong one.                                                                | Raising `ValueError` when `cdf(min) == cdf(max)` is cheaper but rejects valid far-tail requests.                          |
| D5  | Remove `startorch.periodic` (`SineWavePeriodic` with a sampled period is not reproduced)       | **Yes**, unless `SineWavePeriodic` is used by a downstream project; then keep one `PeriodicSineWave(period, cycles, ...)` class.                                              | Keeping the package keeps 616 lines for 5 classes that appear in 5 test files and in no documentation or notebook.                                  |
| D6  | `make_linear_regression` noise on the target; `Hypercube` documented as one-hot plus noise     | **Yes** for the noise (matches scikit-learn and Friedman). Keep the `Hypercube` name and fix the docstring.                                                                   | A `feature_noise_std` option keeps the errors-in-variables behaviour for who needs it.                                    |
| D7  | One `Unary(op)` transformer with a table of names                                              | **Yes.** The names are validated and listed in the error and in the docs. A table is easier to maintain than 25 classes of 8 lines.                                           | Keeping classes gives a named type per function in the config (`_target_`), at 25 times the code.                        |
| D8  | Replace `objectory` by `coola.factory`                                                         | **Not now.** `coola 2` is not installed and its API was not verified in this review. `Configurable` isolates the factory in one place so the change is one file later.        | Doing it first couples the redesign to a dependency upgrade.                                                             |
| D9  | `DEFAULT_SEED` for the datasets, or require the seed                                           | **`DEFAULT_SEED`** (current behaviour with one named constant).                                                                                                              | Requiring the seed is more explicit but changes every call.                                                              |
| D10 | `dtype` / `device` as arguments of `generate`                                                  | **No.** The device follows `rng` and the dtype follows `torch.get_default_dtype()`: no new argument on 80 classes.                                                           | An explicit argument is more flexible but touches every signature and every composite.                                    |

## What this document does not change

- The idea of composing generators and the names of the concepts that users know (`Normal`,
  `Uniform`, `Add`, `Constant`, `SineWave`, `Arange`, `Wiener`, `Markov`, `AutoRegressive`,
  `Periodic`).
- The required and optional dependencies, the lint, type and import rules, the workflow and release
  rules of `CLAUDE.md`.
- The plotting libraries and the dataset functions (`make_*`), apart from the fixes listed.
- The statistical content of the datasets (Friedman, blobs, circles, moons, swiss roll).

## What this review did not do

- It did not run a statistical test of every generator: only the truncated normal was probed, plus
  the bugs of section 2. The 6,014 lines of distribution code were read by pattern; a defect in one
  of the copy-pasted variants could have been missed, which is the argument for the table of laws.
- It did not prototype `Choice`, `Periodic`, `Markov` or the example generators; section 6 says so.
- It did not verify `coola 2`, `objectory` replacement, `batchtensor` upgrades or CUDA behaviour
  (no GPU, `scipy` not installed).
- It did not measure the line and class counts of the target; they are estimates.
