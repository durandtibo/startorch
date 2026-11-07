# Codebase review and suggestions

Scope: all of `src/startorch` (147 files, about 20k lines), plus the tests, packaging, CI, docs
and notebooks.
The code was read with docstrings stripped, so the focus is on logic and structure.
Breaking changes are allowed, so some suggestions are breaking.

## Current state

- `ruff check`, `ty check` and `lint-imports` are clean. 5,800 tests pass in about 7 s.
  Line coverage is 99.9%. The one uncovered line is `integration/iden/data/generator/example.py:84`, which is removed by step 1 of section 7.
- The code is consistent. Every generator has the same shape (`__init__` validation, `__repr__`,
  `generate`) and the docstrings and tests are thorough.
- Most of the weaknesses are structural: repetition, an API that is not uniform, and a few real bugs.
  Tests check shapes and exact values but not the statistics of the output, so they could not catch
  those bugs.

Items are ordered by priority inside each section. Section 1 (bugs) is the most important.
Items marked **[verified]** were reproduced by running code.

---

## 1. Bugs and correctness problems

### 1.1 [verified] `make_markov_chain` gives every batch row the same first state when `init` is 1-D

`sequence/markov.py:144-151`.

- `MarkovChainSequenceGenerator.create_from_generators` builds `init` with `size=(n,)`.
  `make_markov_chain` then calls `torch.multinomial(init, num_samples=1)` on that 1-D tensor.
  The result has shape `(1,)` and is broadcast into `out[:, 1]`, so all rows start in the same state.
- Reproduced: `make_markov_chain(8, 1, T, torch.ones(3))[:, 0]` returned `[2, 2, 2, 2, 2, 2, 2, 2]`.
  With `init=None` the first states differ, as expected.
- Fix: if `init.ndim == 1`, `expand(batch_size, -1)` it before sampling. Also validate that `init`
  is `(n,)` or `(batch_size, n)` and that `transition` is square.
- Add a regression test that checks the first states are not all identical for a large batch.

### 1.2 [verified] `make_sparse_uncorrelated_regression` ignores `noise_std`

`example/sparse_uncorrelated.py:68-76`.

- `noise_std` is validated and then never used: the noise is hard-coded as `std=torch.ones(...)`.
- Reproduced: the residual std is 1.007 for `noise_std=0.0` and for `noise_std=100.0`.
- Fix: use `std=noise_std`, or add the noise only `if noise_std > 0` as the Friedman generators do.
  This is a behaviour change, so the default should probably be changed deliberately and mentioned in
  the changelog.
- This is also a gap in the tests: there is no test that `noise_std` has an effect.

### 1.3 [verified] In-place operations on generator output corrupt stored and cached data

- `AddSequenceGenerator`, `AddScalarSequenceGenerator`, `MulSequenceGenerator`,
  `MulScalarSequenceGenerator` and `FmodSequenceGenerator` (`sequence/math.py:110,157,406,485,532`),
  their tensor equivalents (`tensor/math.py:103,149,358,431,477`), and `mix2sequences`
  (`timeseries/utils/mixed.py:49-50`) all mutate the tensor returned by a child generator.
- `VanillaSequenceGenerator`, `VanillaTimeSeriesGenerator`, `VanillaExampleGenerator` and
  `CacheExampleGenerator` return views or the cached object itself, not copies.
- Reproduced: `AddScalar(Vanilla(zeros), value=1.0)` called twice leaves the stored data at `2.0`.
  `Add([Vanilla(d), Vanilla(ones)])` modified `d` too.
- Fix options: (a) use out-of-place ops in all combinators, which is the simplest and safest;
  (b) make Vanilla/Cache return `.clone()` by default; (c) both. At minimum, document the aliasing
  contract on `BaseXGenerator.generate`: "the returned tensor may be modified in place by the caller"
  or "the returned tensor must not be modified".
- Related: `CacheExampleGenerator` returns the same `dict` object, so adding a key to the result
  changes the cache. Also, `deepcopy=False` is the default and is the dangerous choice.

### 1.4 [verified] Truncated distributions collapse to the bound in the far tail

`random/bounded.py` (`rand_trunc_normal`, `trunc_normal`, `*_cauchy`, `*_log_normal`, ...).

- The implementation is `icdf(uniform(cdf(min), cdf(max))).clamp(min, max)`.
  When `min` and `max` are far in the same tail, `cdf(min) == cdf(max) == 1.0` in float32.
  `icdf(1.0)` is `inf`, and the clamp hides it by returning `max_value`.
- Reproduced: `rand_trunc_normal((5,), mean=0, std=1, min_value=9, max_value=10)` returned
  `[10, 10, 10, 10, 10]`. The correct output should be spread over [9, 10].
- Fix: for the right tail, work with the survival function (`1 - cdf`) or use symmetry.
  `torch.special.ndtr`/`ndtri`/`log_ndtr` give stable results. Alternatively, sample with float64
  internally, or raise an error when `cdf(min) == cdf(max)`. Tail-collapse should never be silent.
- The `.clamp(...)` calls on the result also hide numerical problems elsewhere. Prefer asserting
  that no clamping is needed in tests.

### 1.5 `linear regression`: noise is added to the features, not the targets

`example/regression.py:185-191`. `targets` is computed from the clean features, then the noise is
added to `features`. The target is then not an exact function of the returned features, which is
errors-in-variables regression. This differs from scikit-learn and from the Friedman generators in
the same package, where the noise is added to the target. If this is deliberate, state it clearly in
the docstring and the argument name (`feature_noise_std`). If not, add the noise to the target.

### 1.6 Smaller correctness issues

- **`bool` accepted as `int`.** `check_integer_ge(True, ...)` passes (verified). `isinstance(True, int)`
  is true. coola 2's `validate_int` rejects booleans, so this is solved by step 7.2 (or exclude `bool`
  explicitly in `check_integer_ge`, `check_interval` and `check_std` if the local helpers stay).
- **`check_std` message is wrong.** It says "greater than 0" but accepts 0 (`utils/validation.py:156`).
  The same inconsistency exists in `RandWienerProcessSequenceGenerator` and `wiener_process`
  (`step_size < 0` with the message "greater than 0"). Meanwhile every `rand_*` function rejects
  `std == 0` (`std <= 0`). Pick one rule: either allow zero everywhere (a degenerate distribution is
  useful as a constant noise-free baseline) or nowhere, and fix the messages.
- **Division by zero.** `PeriodicSequenceGenerator`, `PeriodicTimeSeriesGenerator`,
  `RepeatPeriodic*` and `SineWavePeriodic*` compute `math.ceil(seq_len / period)` with a sampled
  `period`. `period == 0` raises a bare `ZeroDivisionError` at generate time. Validate `period >= 1`
  with a clear message.
- **`rand_log_uniform` / `log_uniform` do not validate `low > 0`.** `rand_log_uniform(low=0)` raises a
  cryptic `ValueError: expected a positive input, got 0.0` (verified) and the tensor version returns
  `nan`. The generator classes `RandLogUniform*` do not check it either.
- **`Friedman2/3` can divide by zero.** `features[:, 0]` is drawn from `[0, 100)` and used as a
  denominator in `make_friedman3_regression`. `rand_uniform` can return exactly 0.
- **`NormalizeTransitionGenerator` defaults to `p=2.0`.** For a Markov transition matrix the rows
  should sum to 1, i.e. `p=1`. L2 normalisation gives unit-norm rows, which is not a stochastic
  matrix (`multinomial` hides it by renormalising). Default to `p=1`, or rename the class and doc.
- **`__repr__` omissions.** `MulScalarTensorGenerator.__repr__` and the sequence equivalent leave
  out `value`. `DivTensorGenerator` / `DivSequenceGenerator` leave out `rounding_mode`.
  `AddScalar`'s repr includes the value, so the omission is inconsistent. Generating `__repr__`
  automatically (see 2.3) would remove this class of bug.
- **In-place add/mul on integer tensors.** `AddScalar(RandInt(...), value=0.5)` fails with a dtype error
  because `tensor.add_(0.5)` cannot cast a float into a long tensor. `AddTransformer` and
  `MulTransformer` use `value += ...` on a clone with the same limitation (`transformer/arithmetic.py:99,338`).
  Out-of-place ops promote correctly. This is another reason for fix 1.3.
- **Mid-batch sharing.** `MultinomialChoice*`, `Cat2SequenceGenerator`, `PeriodicSequenceGenerator`,
  `AutoRegressiveSequenceGenerator` and `TimeSeriesExampleGenerator` draw one random choice
  (`order`, `period`, `changepoint`, `seq_len`, which generator) for the entire batch.
  For a synthetic-data library, users likely expect per-example variation for the choice generators
  and the changepoint. Either offer a per-example option or document that clearly in the class docstring.
  Per-example choice also removes the "all examples of a batch come from the same mixture component"
  surprise.
- **`AutoRegressiveSequenceGenerator` wraps with `fmod`.** `torch.fmod(..., max_abs_value)` keeps values
  bounded but introduces discontinuities that are not part of an AR process. A `clamp` or `tanh`-style
  squashing, or a stability check on the coefficients, would be more faithful. The Python loop over
  `seq_len + order * warmup` is also the slowest generator; a `torch.compile`d or
  `scipy.signal.lfilter`-style implementation could help if long sequences matter.
- **`RuntimeError` for bad values.** `scale_batch`, `check_integer_ge`, `check_interval`, `check_std`,
  `str_weighted_modules`, `VanillaSequenceGenerator`, blobs (`centers`/`cluster_std` shape),
  `hist_feature` and others raise `RuntimeError` for invalid arguments (23 sites). The rest of the
  library (119 sites) uses `ValueError`. `RuntimeError` is for runtime failures. See 3.2.
- **Typo.** `rand_trunc_log_normal` names its message variable `mag` instead of `msg`
  (`random/bounded.py:522`). Cosmetic, but it shows the validation code was copy-pasted.
- **`hypercube` is one-hot.** `make_hypercube_classification` puts a `1` on coordinate `target` and adds
  Gaussian noise. That is a one-hot encoding plus noise, not a hypercube-vertex construction like
  scikit-learn's. Rename or document it.
- **Plotting multi-feature sequences.** `plot_sequence` (matplotlib and plotly) calls
  `select_along_batch(batch, i).flatten()`. With `feature_size > 1` this interleaves the features into a
  single zig-zag line instead of one line per feature.
- **Small TODO.** `LinearRegressionExampleGenerator.create_uniform_weights` has a
  `TODO: add missing documentation` and no docstring (`example/regression.py:122`).

---

## 2. Structure and duplication (the largest maintenance cost)

### 2.1 One `setup_*` / `is_*_config` function pair copied 10 times (rename `setup_*` to `resolve_*`)

There are 10 `setup_*` functions (tensor, tensor transformer, sequence, transformer, example,
timeseries, periodic sequence, periodic timeseries, transition, ...) that differ only in the base
class and the log message. Each has the same flaw: if the object is not an instance of the base
class, it only logs a **warning** and returns it, so a wrong type fails much later with an unrelated
`AttributeError`.

- Suggestion: add one generic helper, for example `startorch.utils.factory.resolve_object(obj, base, name)`,
  and make each function a one-line wrapper (or drop them in favour of the helper). Rename the `setup_*` functions to `resolve_*` (`resolve_tensor_generator`, `resolve_sequence_generator`, ...), since they return an existing instance or build one from a config. This is breaking: keep `setup_*` as deprecated aliases for one release.
- Raise `TypeError` for a wrong type instead of logging a warning. This is a (small) breaking change
  that moves the error to construction time.
- The `info` log "Initializing a ... from its configuration" is noisy for large configs. Use `debug`.

### 2.2 The sequence, tensor and distribution classes are a Cartesian product

- For each of the ~11 distributions there are up to 4 classes per level: `X` (parameters come from
  generators), `RandX` (scalar parameters), `TruncX`, `RandTruncX`. These exist for tensors **and**
  for sequences. In total, `sequence/` and `tensor/` hold 170 classes in about 6,000 lines, and most
  differ only in the name of the function they call.
- `RandNormalSequenceGenerator` is exactly `TensorSequenceGenerator(RandNormalTensorGenerator(), feature_size)`,
  and `TensorSequenceGenerator` already exists. All `Rand*SequenceGenerator` classes could be thin
  aliases or removed.
- The same applies to unary maths: `AbsTensorGenerator`, `AbsSequenceGenerator`, `AbsTensorTransformer`,
  and the matching `Exp`, `Log`, `Sqrt`, `Cosh`, ... are each implemented 3-4 times (tensor generator,
  sequence generator, tensor transformer, plus transformer-with-keys in `transformer/`).
  `TransformTensorGenerator(gen, AbsTensorTransformer())` already composes them.
- Suggestion: a single generic `Unary(fn_name)` / `Elementwise` transformer plus a registry, with the
  existing names kept as thin subclasses or aliases for compatibility. A similar generic `Distribution`
  generator, parametrised by a distribution name, would remove thousands of lines.
- Even without a redesign, the `float(x)` / `<= 0` validation and the `__repr__` of the `Rand*` classes
  could come from a small shared base class.

### 2.3 Hand-written `__repr__` and constructor boilerplate (187 `__repr__` methods)

Almost every class stores its constructor arguments in `self._x` and repeats them in a hand-written
`__repr__`. Bugs from this are already present (section 1.6). Options: a base-class `__repr__` driven by
`inspect.signature(self.__init__)`, or `dataclass(frozen=True)`-style classes with `__post_init__`
validation. `objectory` instantiates by `_target_` + kwargs, so both approaches are compatible.

### 2.4 Parallel hierarchies: tensor / sequence / timeseries / example / transformer

The same concepts are re-implemented at each level: `Vanilla`, `Concatenate`, `MultinomialChoice`,
`Transform`, `Periodic`/`Repeat`, `Cache`. `Vanilla{Sequence,TimeSeries,Example}Generator` are almost
identical (same checks, same error messages). `Periodic` logic (`ceil(seq_len/period)`, `repeat`, `slice`)
exists in `sequence/periodic.py`, `timeseries/periodic.py`, `periodic/sequence/repeat.py` and
`periodic/timeseries/repeat.py`.

- A common base (or a mixin) for the "choose one of N weighted generators", "vanilla data slice" and
  "periodic repeat" behaviours would remove most of the copies.
- `startorch.periodic` could be folded into `sequence` / `timeseries` (it is only 8 files and each
  base class has only 1-2 implementations).

### 2.5 Naming collisions and confusing names

- There are two different classes called `BaseTensorTransformer`: `startorch.tensor.transformer.base`
  (tensor to tensor) and `startorch.transformer.base` (a keyed `dict` transformer with an `_transform`
  hook). `setup_tensor_transformer` / `setup_transformer` add to the confusion.
  Rename the second one (for example `BaseKeyTransformer` or `BaseSingleInputTransformer`).
  The package names `startorch.transformer` and `startorch.tensor.transformer` are also easy to mix up.
- `ConcatenateTimeSeriesGenerator` and `ConcatenateExampleGenerator` do not concatenate along any axis:
  they `dict.update` the outputs of several generators (a union of keys). `Merge` and `Cat2` do the real
  concatenation/merge. Rename to `Union`/`Join`, and **fail on duplicate keys** (today `out |= ...`
  silently overwrites).
- `Cat2SequenceGenerator` is hard-coded to two parts. A general `ConcatSequenceGenerator` with N
  generators and N-1 change points would cover it.
- `MixedTimeSeriesGenerator` / `mix2sequences` is not self-describing (it swaps the odd positions of two
  sequences). Give it a descriptive name and document it in the module.
- `generate_uniform_rate` on `PoissonSequenceGenerator` is the only classmethod factory not named
  `create_*`.
- `TimeSeriesExampleGenerator(generators=...)` takes a single generator under a plural name.
- `rng` is the parameter name on `generate(...)`, but `generator` on the `startorch.random` functions and
  the `make_*` functions. Choose one.
- Every class is exported twice (`Abs` and `AbsSequenceGenerator`, `Normalize` and
  `NormalizeTransitionGenerator`, ...). This doubles the public surface and the docs. Keep one
  canonical name and mark the other as deprecated, or drop the short one.

### 2.6 Inconsistent signatures

- Keyword-only is applied to some functions and not to others: `rand_trunc_cauchy(size, *, loc, ...)` but
  `rand_cauchy(size, loc, scale, generator)`; `trunc_exponential(rate, max_value, generator)` but
  `trunc_cauchy(*, loc, ...)`; `rand_poisson(size, rate, generator)`; `wiener_process(seq_len, step_size,
  batch_size, generator)` where `seq_len` comes before `batch_size` but the generator methods use
  `(seq_len, batch_size)`. Make every option after the first required argument keyword-only.
- `make_blobs_classification(num_examples, centers, *, ...)`, `make_linear_regression(weights, *, ...)` and
  the other `make_*` functions do not take their main argument the same way.
- `startorch.random` has no `poisson(rate_tensor, generator)` although every other distribution has the
  `rand_x` / `x` pair; `PoissonTensorGenerator` calls `torch.poisson` directly.
- In `startorch/random/semi_infinite.py`, `from startorch.random import cauchy, normal, ...` imports from
  the package `__init__`. It works only because of import order. Import from `startorch.random.infinite`.

---

## 3. API design

### 3.1 No `dtype` / `device` control

There is no mention of `device` or `dtype` in `src` apart from `seed.py` and `Full*`. Everything is
generated as float32 on CPU. As a consequence:

- A CUDA `torch.Generator` cannot be used (random ops are called without `device=`).
- float64 data, or float16/bfloat16 for model testing, requires a post-hoc `.double()`.
- This also contributes to the tail problem in 1.4 (float64 would make it much less likely).
- Suggestion: add `dtype` and `device` keyword arguments (default `None` = current behaviour) to
  `rand_*` and the base `generate`, or a context manager that sets them. At the very least, document
  that generators are CPU/float32 only.

### 3.2 Exceptions

Use `ValueError` (bad value) and `TypeError` (bad type) for argument validation. Keep `RuntimeError` for
runtime failures. Consider one project exception hierarchy only if a user needs to catch them
(probably not). See the list in 1.6.

### 3.3 Reproducibility and the RNG contract

- `rng=None` means "use the global torch RNG". That is a reasonable default but the docs should say so.
- Document, and test, the guarantee: the same seed gives the same output across runs, and the number of
  random draws of each generator does not change when an unrelated option changes. The order in which
  composite generators draw (`low` then `high`, ...) is part of the reproducibility contract, so a
  refactor like 2.2 needs golden-value tests (see 5.2).
- Magic seeds are scattered: `13683624337160779813` (4 plot defaults), `17532042831661189422`
  (blobs, regression), and others. Define `DEFAULT_SEED` in `startorch.constants` (or require the seed).
- `ExampleDataGenerator.equal` compares `initial_seed()`, not the current RNG state, so two generators
  that have produced different numbers of batches compare equal.

### 3.4 Typing

- `pyproject.toml` has the classifier `Typing :: Typed` but `src/startorch/py.typed` does not exist, so
  type checkers will treat the package as untyped for users. Add the file.
- `data: dict[Hashable, torch.Tensor]` is used for keys but several constructors annotate keys as `str`.
- `size: tuple[int, ...]` on `generate` but `list[int] | tuple[int, ...]` on the random functions. Use
  `Sequence[int]` or a shared alias such as `startorch.typing.Size`.
- The tensor helpers use `torch.Tensor | Sequence[float]`, `feature_size: tuple[int, ...] | list[int] | int`
  in about 70 places. A shared alias (`FeatureSize`) would shorten signatures and ensure `to_tuple` is
  always applied.

### 3.5 Validation and error messages

- Several validations happen only at `generate` time (`order < 1`, `period == 0`, `num_mask`,
  `init`/`transition` shape). Fail in `__init__` when the argument is a constant.
- Error messages are inconsistent in format ("has to be", "must be", "Expected", "Incorrect value for").
  A central helper set in `startorch.utils.validation` (`check_positive`, `check_non_negative`,
  `check_less_equal`, `check_same_shape`, `check_in_interval`) would also remove the ~60 repeated
  `if torch.any(x.le(0.0))` blocks in `random/bounded.py`.
- `random/bounded.py` (1,038 lines) repeats the pattern: validate shapes, validate positivity, validate
  order, compute CDF bounds, sample, clamp. Rewriting `rand_X(size, ...)` as
  `X(full(size, ...))` and having one scalar-parameter path would remove half of the file.
- `prepare_probabilities` does not reject `nan`/`inf`, and `weights.min()` fails on an empty tensor with
  an unhelpful error.
- `mask_by_row` loops over rows with `torch.randperm`; `torch.rand(n_rows, n_cols).argsort(dim=1)[:, :n]`
  is vectorised and draws from the generator once. `mask_square_matrix` can use `masked_fill` instead of
  `masked_scatter` with a full-size source tensor. `circulant` fails on an empty vector (`length=-1`).
- `to_array` / `hist_feature` (matplotlib) call `.numpy()` directly, which fails on tensors that require
  grad or live on a GPU. Use `.detach().cpu().numpy()`. `hist_feature` in the matplotlib module does not
  use `to_array` while the plotly one does.

### 3.6 Missing features worth considering

- Per-example mixtures (see 1.6) and conditional generators, for example "class label, then features
  drawn from a class-specific distribution". Today `Blobs` is the only one.
- A seasonal+trend+noise time-series builder that is a first-class object. It is what most users of a
  "synthetic time-series generator" would build with `Add`, `Linear`, `SineWave`, and `Normal`.
- Missing-value / irregular-sampling / outlier injection transformers. `Merge` and `Time` cover part of
  irregular sampling.
- A `poisson` tensor function (2.6) and the other distributions that `torch.distributions` provides
  (Beta, Gamma, Student-t, Laplace, Bernoulli, ...). With a generic distribution adapter (2.2) these
  would be nearly free.
- `ArangeSequenceGenerator` has no `start`/`step`/`dtype`.
- `RandWienerProcessSequenceGenerator` supports only one feature (it hard-codes `unsqueeze(dim=2)`) while
  most other `Rand*` sequence generators take `feature_size`.
- Plotting only exists for sequences and feature histograms. Plots for `TimeSeries` and `Example`
  outputs (scatter for the 2-D classification datasets) would help the docs and notebooks.
  matplotlib and plotly modules are mirror images; a shared data-preparation layer would avoid drift
  (for example, `plot_sequence` has `xscale`/`yscale` in matplotlib but not in plotly; the `figsize`
  meaning differs between the two `hist_feature`s).

---

## 4. Performance

- `make_markov_chain` and `AutoRegressiveSequenceGenerator` use Python loops over the sequence length
  (inherent for Markov chains, but the AR loop can be written as a convolution/`lfilter` when there
  is no `fmod`). `mask_by_row` loops over rows (3.5).
- `blobs` loops over clusters and builds `expand`ed views for every cluster. A single `normal(...)` over
  `(num_clusters, per_cluster, feature_size)` is simpler and faster.
- `ConstantSequenceGenerator` uses `repeat_along_seq`, which allocates. An `expand` would be free until
  the tensor is modified (but see 1.3 about aliasing: that is a reason to keep `repeat` unless the
  combinators become out-of-place).
- Composite generators evaluate every child eagerly even when an argument is a constant
  (`UniformTensorGenerator(low=Full, high=Full)` samples and stacks full tensors). Probably fine, but a
  scalar fast-path would speed up the most common configurations.
- `Cache`/`Vanilla` slicing is O(1), good. `hist_sequence` default `batch_size=10000`, `seq_len=1000`
  generates 10M values per call.

---

## 5. Tests

The suite is large and clean (one assert per test, exact values, error-message matching). Gaps:

### 5.1 No statistical tests

Zero uses of `kstest` / `ks_2samp`. All distribution tests check shape, range, dtype and, sometimes,
a mean. That cannot catch problems like 1.4 (the tail collapse) or a wrong scale parametrisation.
Add a small `tests/integration/test_distributions.py` that, with a fixed seed and `n=10_000`, runs a KS
test against `scipy.stats` (or a closed-form CDF) for each `rand_*` function. A marker for "statistical"
tests is also useful so a seed change can be investigated.

### 5.2 No golden-value tests across versions

Refactoring suggestions in section 2 change the order of random draws. Store a handful of
`seed -> output` fixtures per generator family, so an unintended reproducibility break is visible
in review.

### 5.3 Very few integration tests

`tests/integration/` has only `test_smoke.py`, which tests one generator (`CirclesClassification`).
`CLAUDE.md` says each patched unit test needs a real counterpart. A parametrised smoke test that
builds each generator from a `_target_` config dict, calls `generate`, and checks shape/dtype/finite
values would catch registration, config and wiring errors for all ~170 classes. A "round-trip repr"
test (build from config, `repr`) would have caught the missing fields in 1.6.

### 5.4 Missing regression tests for the bugs above

1.1 (Markov `init`), 1.2 (`noise_std`), 1.3 (aliasing), 1.4 (tails). Each must fail without the fix.

### 5.5 Smaller points

- One uncovered line: `integration/iden/data/generator/example.py:84`. It goes away when the integration package is removed (7.0).
- Add property-based tests (Hypothesis is already in `.gitignore`/`clean`) for the validation helpers
  and shape invariants: output shape equals `(batch_size, seq_len, *feature_size)` for every sequence
  generator, `seq_len=0` and `batch_size=0` do not crash where documented, etc.
- Edge cases to cover for composites: `seq_len=0` for `Cat2` (one half has zero length) and for AR.
- Thread-safety and "does not mutate input" tests exist for some helpers; add them to the combinators
  after 1.3.

---

## 6. Tooling, packaging, CI and docs

- **`py.typed`** is missing (3.4).
- **`startorch.__init__`** is empty. There is no `__version__` and no top-level API. Adding
  `__version__ = importlib.metadata.version("startorch")` is cheap and standard.
- **Version.** `0.3.0a0` with `batchtensor >=0.1.2a1` (and `iden >=0.2.1a1`, removed in 7.0) as alpha pins. Check that a
  release can be installed from PyPI without `--pre` for the optional extras.
- **Python versions.** Classifiers list 3.15 while the lock/CI environment runs 3.14. Verify CI covers
  all listed versions or trim the list.
- **Both `pyright` and `ty`** are dev dependencies and both are configured, but `CLAUDE.md` and CI use
  only `ty`. Drop the unused one, or run both.
- **Stray directories** in the working tree: `benchmark/` (tracked) and `benchmarks/tmp` (untracked),
  `coverage.xml`, `htmlcov/` and `dist/` (ignored). Merge to a single `benchmarks/` or remove the
  untracked one. `benchmark/shard_loading.py` imports `iden`; both are removed in 7.0.
- **Notebooks.** `notebooks/loss.ipynb` and `notebooks/plot_plotly.ipynb` are untracked. Four tracked
  notebooks (`sequence_core`, `sequence_advanced`, `example`, `async_time_series`) are not executed in
  CI, so they can silently break after an API change. Run them with `nbmake` or `jupyter nbconvert
  --execute` in a nightly job, or convert key ones into docs pages.
- **Docs.**
  - The "User Guide" has one page (`sequence.md`) and the `quickstart.md` entry is commented out in
    `mkdocs.yml`. Missing guides: tensors, transformers, time series and example generators, config-driven
    construction (`_target_`), reproducibility/RNG, writing a custom generator for each level.
  - `sequence.md` lists generator names by hand ("non-exhaustive list"), which will drift. Generate it
    from the package `__all__`, or link to the reference.
  - There is no changelog file; the project links to GitHub releases. Add `CHANGELOG.md` (CLAUDE.md asks for
    tracking documents to be updated with the PR number).
  - No docs page for `startorch.random` semantics (what "Rand" vs plain, "Trunc" mean). A naming-convention
    table would remove a lot of confusion (2.2).
- **CI.** 33 workflow files. Many are `bot-*` and shared-Makefile syncs. The set is thorough
  (type checking, cyclic imports, doctest, security, nightly dependency matrix). Suggestions: add a
  job that runs the integration tests with coverage thresholds (`--cov-fail-under=100`, currently
  99.9%), and a job for the notebooks (above).
- **Ruff.** The configured rule set is already strict. `C90` complexity max 10 and `PLR0913` ignored
  (too many arguments) is a reasonable trade-off. Consider enabling `ANN` for tests only if desired,
  and `PLR2004` ignored globally hides magic numbers such as the seeds in 3.3.

---

## 7. Suggested order of work

1. **Remove `startorch.integration` and the `iden` dependency** (planned, not done yet; see 7.0).
   Do it first: it deletes code and a dependency that the later steps would otherwise have to migrate.
2. **Upgrade coola to `>=2.0.0a2,<3.0`** (a prerequisite for several items below).
   See section 7.1 for what the upgrade requires and what it makes possible.
3. **Reduce duplication by relying on coola** (see section 7.2). Do it right after the
   upgrade, because it changes imports in nearly every file and later PRs would otherwise conflict.
4. **Clean the codebase of unused code** (see 7.3). Do it once the coola de-duplication has removed
   the local helpers, so the dead-code report only lists what is really left.
5. **Bug-fix PRs** (small, independent, each with a regression test): 1.1 Markov `init`;
   1.2 `noise_std` in sparse-uncorrelated; 1.6 repr omissions; 1.6 `period == 0` and `low <= 0`
   validation. (The `bool` accepted as `int` bug in `check_integer_ge` disappears with 7.2, because
   coola's `validate_int` rejects booleans.)
6. **Aliasing** (1.3): switch combinators to out-of-place ops; clone in `Vanilla`/`Cache` (or
   document); add tests.
7. **Truncated tails** (1.4): fix, then add the statistical tests (5.1).
8. **Cheap, mechanical:** `py.typed`, `__version__`, `DEFAULT_SEED`, `TypeError` in the `resolve_*`
   functions, `ValueError` instead of `RuntimeError` (2.1, 3.2).
9. **Integration/golden tests** (5.2, 5.3), before any large refactor.
10. **Structural refactors** (2.2-2.5): generic elementwise and distribution adapters, auto `__repr__`,
   renaming the colliding classes, merging `periodic` into `sequence`/`timeseries`. These are
   breaking, so do them in one release (`0.3.0` is still an alpha) with deprecation aliases for the
   old names.
11. **Features and docs** (3.1 dtype/device, 3.6, section 6): after the structure is simplified.

### 7.0 Remove `startorch.integration` and `iden`

Decision: drop the optional `iden` integration. Nothing is removed by this review; this is the plan for
a dedicated PR (a breaking change, mentioned in the PR description):

- Delete `src/startorch/integration/` (`ExampleDataGenerator`, `fallback_types.py`) and
  `tests/unit/integration/`.
- Delete the `iden` helpers in `startorch/utils/imports.py` (`check_iden`, `is_iden_available`,
  `iden_available`) and `iden_available` in `startorch/testing.py`, with their tests in
  `tests/unit/utils/test_imports.py`.
- Delete `benchmark/` (`shard_loading.py` is the only file; it needs `iden` and the integration package).
- `pyproject.toml`: remove the `iden` extra and `"startorch.integration"` from both import-linter
  contracts. Refresh `uv.lock`, and remove `iden` from `dev/config/package_versions.json`.
- Docs: remove `iden` from the README dependency table and from `CLAUDE.md` (optional libraries and the
  `iden_available` marker).
- Keep `tests/integration/` and the CI integration-test jobs: they are unrelated.
- Verify with `ruff`, `ty`, `lint-imports`, the unit tests and `inv test.doctest`. The earlier uncovered
  line (see "Current state") disappears with this change.
- Side effects on this review: items that mention `iden` (the `batchtensor`/`iden` pin check in 7.1,
  the `iden_available` row in 7.2, the stray `benchmark/` directory, the alpha pins) become moot.

### 7.1 Upgrade to `coola>=2.0.0a2,<3.0`

`pyproject.toml` currently pins `coola >=0.9.1,<2.0`. I unpacked the `2.0.0a2` wheel (read only, not
executed) to check what changes. It is a reorganisation, so this is **not** only a version bump:

- `coola.utils` no longer exports anything (`__all__` is empty). Every
  `from coola.utils import str_indent, str_mapping, ...` and `from coola.utils.format import ...`
  in `src` (about 60 import lines in 5 spellings) must change to `coola.formatting.structure`
  (`str_indent`, `str_mapping`, `str_sequence`, `repr_indent`, `repr_mapping`, `repr_sequence`; there
  are also `*_line` variants). `num_spaces` is keyword-only in 2.0, which `startorch` already uses.
- `objectory` is re-implemented as `coola.factory` (`factory`, `resolve_object`, `instantiate_object`,
  `is_object_config`, `OBJECT_TARGET`). See 7.2: it can replace the `objectory` dependency too.
- `coola.utils.imports` is now a package. `package_available` is in
  `coola.utils.imports.package`; `decorator_package_available` is not found under that name in 2.0.0a2,
  so `startorch.utils.imports` needs a small adaptation or a move to coola's own `is_*_available` /
  `check_*` helpers.
- `objects_are_equal` is still in `coola.equality` (check the import path in the code that uses it).
- The packages `batchtensor` and `objectory` (and `iden` if it is still there) pin `coola` themselves. Check that
  `batchtensor >=0.1.2a1` resolves with coola 2 (an alpha of those may be needed),
  and refresh `uv.lock`.
- Docs and tests: run the doctests (`inv test.doctest`) because `repr` / `str` output of nested
  structures could differ.
- Several items in this review become simpler or disappear after the upgrade (see 7.2). The PR should
  contain only the upgrade, the import changes and the lock file, with no behaviour change.

### 7.2 Reduce duplication: use coola instead of re-implementing it

coola 2 provides many of the helpers that `startorch` re-implements. Replace them and delete the local
copies and their tests. Each row is one PR-sized change.

| startorch code | coola 2.0 replacement | Notes |
| --- | --- | --- |
| `utils/validation.py`: `check_integer_ge`, `check_interval`, `check_std`, `check_feature_size`, `check_num_examples` | `coola.validation`: `validate_int`, `validate_number`, `validate_ge`, `validate_gt`, `validate_range`, `validate_finite`, `validate_isinstance`, `validate_in`, `validate_not_empty` | Fixes the `bool`-as-`int` bug (1.6) and the wrong "greater than 0" message. Keep thin domain wrappers (`check_feature_size`, `check_num_examples`) only if they add meaning. Exception types change: check the tests that match messages. |
| `setup_*` / `is_*_config` (10 copies, see 2.1) | `coola.factory.resolve_object` and `coola.factory.is_object_config` | One helper replaces the 10 copies and can raise instead of warning on a wrong type. Check it accepts an abstract base class as target. |
| `objectory` dependency (`AbstractFactory`, `OBJECT_TARGET`) | `coola.factory` (`factory`, `OBJECT_TARGET`, `instantiate_object`) | Would remove one required dependency. The `metaclass=AbstractFactory` registration of subclasses needs an equivalent; verify before committing to it. Treat as a separate PR after the helper above. |
| `utils/seed.py`: `get_torch_generator`, `get_random_seed` (and `setup_torch_generator`) | `coola.random.make_torch_generator(seed, device=...)`, `coola.random.sample_torch_seed(seed)`, `validate_seed` | Gives `device` support (3.1) for free and removes the local magic-number handling. Keep `setup_torch_generator` (accepts a seed or a generator) as the only local helper, or resolve it with a coola helper if one exists. |
| `utils/conversion.py`: `to_array`, `to_tensor` | `coola.utils.array.to_array`, `coola.utils.tensor.to_tensor` | Also addresses the `.numpy()`-on-grad-tensor problem (3.5) if coola handles it; check. Keep `to_tuple` (or replace by a typed alias, 3.4). |
| `utils/format.py`: `str_weighted_modules` | `coola.formatting.structure` (`str_sequence`, `str_mapping`) | The weighted variant is small; implement it with `str_mapping` over `{"(0)": ..}` or keep as one-liner on top of coola. |
| `utils/imports.py` (312 lines: 5 packages x `is_*_available`, `check_*`, decorator) | `coola.utils.imports` (`package_available`, optional `is_*_available` / `check_*` for torch, numpy, scipy, ...) | coola has the helpers for its own optional packages, not for `matplotlib`, `plotly` (`iden` is removed in 7.0). Generate the 2 remaining families from one small factory function instead of 5 copies of 40 lines (`objectory` goes away if the row above is done). |
| `testing.py`: `matplotlib_available`, ... markers | Same pattern, or `coola.testing` | Check what `coola.testing` provides before keeping a local copy. |
| `utils/tensor.py`: `shapes_are_equal` and the loops in `random/bounded.py` that check same shapes | `coola.equality` / `coola.validation` helpers | Only if one exists for shapes. Otherwise add one `check_same_shape` in `utils.validation` (3.5). |
| `ExampleDataGenerator.equal` and the `objects_are_equal` use (moot after 7.0) | `coola.equality` | Already used; check that the new tester registry removes the need for the hand-written `equal`. |
| `utils/mask.py`, `utils/weight.py` | none | No coola equivalent found. Keep them. |

Process for each row: add the coola call, delete the local function and its unit tests (do not keep a
wrapper that only forwards), run `inv test.doctest`, `lint-imports` and the unit tests of the packages
that import it. Keep `startorch.utils` free of imports from other `startorch` packages. Mention
behaviour changes (exception type, message) in the PR description.

Also look for these before writing any new helper: coola has `coola.nested` and `coola.recursion`
(`apply`-style traversal of nested dicts of tensors, used by `batchtensor`), `coola.reducer`,
`coola.registry`, `coola.rounding`, `coola.hashing` and `coola.iterator`. The "dict of tensors" code in
`transformer/*`, `example/*` and `timeseries/*` (copy, key checks, slicing) may be replaceable by them
or by `batchtensor.nested`.

### 7.3 Clean the codebase of unused code

Goal: delete code that nothing uses, so the later refactors (section 2) have less to move. One PR per
area, no behaviour change, each deletion also removes its tests.

How to find candidates:

- Run a dead-code detector on `src` and review the output by hand (it has false positives for classes
  built from `_target_` strings, for the `__init__` aliases and for public API): `vulture src --min-confidence 80`
  (add `vulture` to the `dev` group, or run it once with `uvx vulture`). Add a whitelist file for the
  intended public names instead of ignoring the tool.
- Ruff already covers unused imports and variables (`F401`, `F841`, `ARG`). Add the `ERA` rule (commented-out
  code) and check `RUF100` (unused `noqa`) to find the leftovers.
- Cross-check each candidate with `grep` in `src`, `tests`, `docs`, `notebooks` and the README before deleting.
  Because `startorch` is a library, an unused public name is a breaking change: remove it in the same
  release as the other breaking changes (section 7, step 10) and list it in the changelog.

Candidates already seen during this review (to confirm):

- `startorch/constants.py`: `SEQUENCE`, `SEQ_LEN` and `BATCH_SIZE` are not used anywhere in `src`
  (`VALUE` and `TIME` are used only 2 and 4 times).
- `transformer/base.py`: commented-out `__repr__` in `BaseTensorTransformer` (dead code, delete it).
- `utils/weight.py`: module constants `GENERATOR` and `WEIGHT`, check whether anything else needs them.
- The duplicate short aliases in the `__init__.py` files (`Abs` next to `AbsSequenceGenerator`, ...), see
  2.5: keep one name per class and delete the other.
- `startorch.utils.imports`: `objectory` helpers (`check_objectory`, `is_objectory_available`,
  `objectory_available`) when `objectory` is a required dependency and cannot be missing; the
  `Mock()` fallbacks in `plot/*` modules.
- Helpers replaced by coola (7.2) and the `iden` code (7.0) are removed in their own steps, not here.
- Stray files in the tree (`benchmarks/tmp`, the two untracked notebooks `notebooks/loss.ipynb` and
  `notebooks/plot_plotly.ipynb` if they are scratch work), and unused CI workflows/bot files if any.
- Docs: stale README dependency tables (very old version ranges such as `python >=3.9`), the
  commented `quickstart.md` nav entry.

Check after each PR: `ruff check`, `ty check`, `lint-imports`, unit tests, `inv test.doctest`, and that
coverage stays at 100% (removing dead code should not lower it).

## 8. What I did not do

- Statistical correctness was checked only for the specific cases described above. I did not run a
  full distribution test of every generator.
- I read the 5 large `random`/`tensor`/`sequence` modules by pattern (they are near-identical), and
  read in full the modules that differ (`ar`, `markov`, `time`, `periodic`, `joining`, `wiener`, `example/*`,
  `timeseries/*`, `transformer/*`, `transition/*`, `plot/*`, `utils/*`). A bug in one of the
  copy-pasted variants that does not appear in the others could have been missed.
- Test files, CI workflow files and the `.make/` shared Makefiles were reviewed only at the
  structure level, not line by line.
- Nothing was changed in `src/` or `tests/`.
