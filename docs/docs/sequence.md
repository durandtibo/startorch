# Sequence

:book: This page is a quick overview of how to use sequence generators, and how to implement custom
sequence generators.
This page does not present the implementation of the builtin sequence generators.

## Introduction

The main objects to generate synthetic sequences are the sequence generators.
The API of the sequence generator is defined in `BaseSequenceGenerator`.
Note that `startorch` generates sequences by batch to be more efficient.
It is not possible to generate a single sequence, but it is possible to generate a batch of one
sequence.

## Builtin sequence generators

`startorch` has a lot of builtin sequence generators.
Below is a non-exhaustive list of the sequence generators:

- `Abs`
- `Acosh`
- `Add`
- `AddScalar`
- `Arange`
- `Asinh`
- `AsinhUniform`
- `Atanh`
- `AutoRegressive`
- `Cat2`
- `Cauchy`
- `Clamp`
- `Constant`
- `Cosh`
- `Cumsum`
- `Div`
- `Exp`
- `Exponential`
- `Float`
- `Fmod`
- `Full`
- `HalfCauchy`
- `HalfNormal`
- `Linear`
- `Log`
- `LogNormal`
- `LogUniform`
- `Long`
- `Mul`
- `MulScalar`
- `Multinomial`
- `MultinomialChoice`
- `Neg`
- `Normal`
- `Poisson`
- `RandAsinhUniform`
- `RandCauchy`
- `RandExponential`
- `RandHalfCauchy`
- `RandHalfNormal`
- `RandInt`
- `RandLogNormal`
- `RandLogUniform`
- `RandNormal`
- `RandPoisson`
- `RandTruncCauchy`
- `RandTruncExponential`
- `RandTruncHalfCauchy`
- `RandTruncHalfNormal`
- `RandTruncLogNormal`
- `RandTruncNormal`
- `RandUniform`
- `RandWienerProcess`
- `SineWave`
- `Sinh`
- `Sort`
- `Sqrt`
- `Sub`
- `Tanh`
- `TensorSequence`
- `Time`
- `TruncCauchy`
- `TruncExponential`
- `TruncHalfCauchy`
- `TruncHalfNormal`
- `TruncLogNormal`
- `TruncNormal`
- `Uniform`
- `UniformCategorical`

These builtin sequence generators can been seen as basic blocks to generate sequences, or to build
more complex sequence generators.

## Generate a batch of sequences

This section shows how to use a sequence generator to generate a batch of sequences.
Let's assume we want to generate a batch of sequences where the value are sampled from a uniform
distribution `U[-5, 5]`.
This can be easily done with `startorch` by writing the following lines.

```python
from startorch.sequence import RandUniform

generator = RandUniform(low=-5, high=5)
print(generator.generate(seq_len=6, batch_size=2))
```

Output:

```textmate
tensor([[[ 2.6437],
         [ 2.1046],
         [ 3.9529],
         [-2.9899],
         [ 3.6624],
         [-3.8132]],

        [[ 1.9843],
         [ 3.1455],
         [ 3.2380],
         [ 1.3003],
         [ 1.0235],
         [-4.7955]]])
```

`seq_len` controls the sequence length and `batch_size` controls the number of sequences in the
batch.

## Combining sequence generators

A lot of the builtin sequence generators and modular and can be combined to build more complex
sequence generators.
The following example shows how to build a sequence generator that sums the outputs of three sine
wave sequence generators.

```python
from startorch.sequence import (
    Add,
    Arange,
    Constant,
    RandLogUniform,
    RandUniform,
    SineWave,
)

generator = Add(
    (
        SineWave(
            value=Arange(),
            frequency=Constant(RandLogUniform(low=0.01, high=0.1)),
            phase=Constant(RandUniform(low=-1.0, high=1.0)),
            amplitude=Constant(RandLogUniform(low=0.1, high=1.0)),
        ),
        SineWave(
            value=Arange(),
            frequency=Constant(RandLogUniform(low=0.01, high=0.1)),
            phase=Constant(RandUniform(low=-1.0, high=1.0)),
            amplitude=Constant(RandLogUniform(low=0.1, high=1.0)),
        ),
        SineWave(
            value=Arange(),
            frequency=Constant(RandLogUniform(low=0.01, high=0.1)),
            phase=Constant(RandUniform(low=-1.0, high=1.0)),
            amplitude=Constant(RandLogUniform(low=0.1, high=1.0)),
        ),
    )
)
batch = generator.generate(seq_len=128, batch_size=4)
```

<div align="center"><img src="https://durandtibo.github.io/startorch/assets/figures/add3sinewaves.png" width="640" align="center"></div>

## Randomness

It is possible to control the randomness of the sequence generators to make the process
reproducible.
A `torch.Generator` object is used to control the randomness.
The following example shows how to generate the same batch of sequences.

```python
from startorch.sequence import RandUniform
from startorch.utils.seed import get_torch_generator

generator = RandUniform(feature_size=())
print(generator.generate(seq_len=6, batch_size=2, rng=get_torch_generator(1)))
print(generator.generate(seq_len=6, batch_size=2, rng=get_torch_generator(1)))
```

Output:

```textmate
tensor([[0.7576, 0.2793, 0.4031, 0.7347, 0.0293, 0.7999],
        [0.3971, 0.7544, 0.5695, 0.4388, 0.6387, 0.5247]])
tensor([[0.7576, 0.2793, 0.4031, 0.7347, 0.0293, 0.7999],
        [0.3971, 0.7544, 0.5695, 0.4388, 0.6387, 0.5247]])
```

The two generated tensors have the same values.

## How to implement a sequence generator

This section explains how to implement a custom sequence generator.
`startorch` has a lot of builtin sequence generator but it is possible to implement custom sequence
generators.
A custom sequence generator has to follow the API defined in `BaseSequenceGenerator`.

Let's assume we want to generate a sequence generator that returns sequence filled with only the
number 42. The following piece of code shows how to implement this sequence generator.

```python
from __future__ import annotations

import torch

from startorch.sequence import BaseSequenceGenerator
from startorch.utils.conversion import to_tuple


class FortyTwoSequenceGenerator(BaseSequenceGenerator):
    def __init__(
        self,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:  # This method is optional but nice to have
        return f"{self.__class__.__qualname__}(feature_size={self._feature_size})"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return torch.full((batch_size, seq_len) + self._feature_size, 42.0)


generator = FortyTwoSequenceGenerator(feature_size=())
print(generator)
print(generator.generate(seq_len=6, batch_size=2))
```

Output:

```textmate
FortyTwoSequenceGenerator(feature_size=())
tensor([[42., 42., 42., 42., 42., 42.],
        [42., 42., 42., 42., 42., 42.]])
```

This implementation allows to configure the feature size.

## Naming conventions

`RandXXX` indicates a standalone sequence generator i.e. a sequence generator that does not
require sequence generators as input to work.
For example, `RandUniform` is the standalone version of `Uniform`.
`Uniform` allows to build more complex sequence generators but it can be more difficult to use
because it has more parameters to configure.

- `RandUniform` example

```python
from startorch.sequence import RandUniform

generator = RandUniform(low=-5, high=5)
generator.generate(seq_len=128, batch_size=4)
```

<div align="center"><img src="https://durandtibo.github.io/startorch/assets/figures/rand_uniform_5.png" width="640" align="center"></div>

- `Uniform` example

```python
from startorch.sequence import RandUniform, Uniform

generator = Uniform(low=RandUniform(-1.0, 0.0), high=RandUniform(0.0, 1.0))
generator.generate(seq_len=128, batch_size=4)
```

<div align="center"><img src="https://durandtibo.github.io/startorch/assets/figures/uniform_1.png" width="640" align="center"></div>


`TruncXXX` indicates the sequence generator sampled value from a truncated distribution.
For example, `RandTruncCauchy` samples values from a truncated Cauchy distribution
whereas `RandCauchy` samples values from a Cauchy distribution.

- `RandCauchy` example

```python
from startorch.sequence import RandCauchy

generator = RandCauchy()
generator.generate(seq_len=128, batch_size=4)
```

<div align="center"><img src="https://durandtibo.github.io/startorch/assets/figures/rand_cauchy.png" width="640" align="center"></div>

- `RandTruncCauchy` example

```python
from startorch.sequence import RandTruncCauchy

generator = RandTruncCauchy()
generator.generate(seq_len=128, batch_size=4)
```

<div align="center"><img src="https://durandtibo.github.io/startorch/assets/figures/rand_trunc_cauchy.png" width="640" align="center"></div>
