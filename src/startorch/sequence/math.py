r"""Contain the implementation of sequence generators that computes
mathematical functions on sequences/tensors."""

from __future__ import annotations

__all__ = [
    "AbsSequenceGenerator",
    "AddScalarSequenceGenerator",
    "AddSequenceGenerator",
    "ClampSequenceGenerator",
    "CumsumSequenceGenerator",
    "DivSequenceGenerator",
    "ExpSequenceGenerator",
    "FmodSequenceGenerator",
    "LogSequenceGenerator",
    "MulScalarSequenceGenerator",
    "MulSequenceGenerator",
    "NegSequenceGenerator",
    "SqrtSequenceGenerator",
    "SubSequenceGenerator",
]


from typing import TYPE_CHECKING

from batchtensor.tensor import cumsum_along_seq
from coola.utils.format import str_indent, str_mapping, str_sequence

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.sequence.wrapper import BaseWrapperSequenceGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class AbsSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that computes the absolute value
    of a generated sequence.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Abs, RandNormal
    >>> generator = Abs(RandNormal())
    >>> generator
    AbsSequenceGenerator(
      (sequence): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).abs()


class AddSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator that adds multiple sequences.

    ``sequence = sequence_1 + sequence_2 + ... + sequence_n``

    Args:
        sequences: The sequence generators or
            their configuration.

    Raises:
        ValueError: if no sequence generator is provided.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Add, RandUniform, RandNormal
    >>> generator = Add([RandUniform(), RandNormal()])
    >>> generator
    AddSequenceGenerator(
      (0): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
      (1): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(self, sequences: Sequence[BaseSequenceGenerator | dict]) -> None:
        super().__init__()
        if not sequences:
            msg = "No sequence generator. You need to specify at least one sequence generator"
            raise ValueError(msg)
        self._sequences = tuple(setup_sequence_generator(generator) for generator in sequences)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._sequences))}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        output = self._sequences[0].generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        for sequence in self._sequences[1:]:
            output.add_(sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng))
        return output


class AddScalarSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that adds a scalar value to a
    generated batch of sequences.

    Args:
        generator: The sequence generator or its
            configuration.
        value: The scalar value to add.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import AddScalar, RandUniform, RandNormal
    >>> generator = AddScalar(RandUniform(), 42.0)
    >>> generator
    AddScalarSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
      (value): 42.0
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        generator: BaseSequenceGenerator | dict,
        value: float,
    ) -> None:
        super().__init__(generator=generator)
        self._value = value

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"sequence": self._generator, "value": self._value}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        sequence = self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        sequence.add_(self._value)
        return sequence


class ClampSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator to generate a batch of sequences
    where the values are clamped.

    Note: ``min_value`` and ``max_value`` cannot be both ``None``.

    Args:
        generator: The sequence generator or its
            configuration.
        min: The lower bound. If ``min_value`` is ``None``,
            there is no lower bound.
        max: The upper bound. If ``max_value`` is ``None``,
            there is no upper bound.

    Raises:
        ValueError: if both ``min`` and ``max`` are ``None``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Clamp, RandNormal
    >>> generator = Clamp(RandNormal(), -1.0, 1.0)
    >>> generator
    ClampSequenceGenerator(
      (sequence): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
      (min): -1.0
      (max): 1.0
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        generator: BaseSequenceGenerator | dict,
        min: float | None,  # noqa: A002
        max: float | None,  # noqa: A002
    ) -> None:
        super().__init__(generator=generator)
        if min is None and max is None:
            msg = "`min` and `max` cannot be both None"
            raise ValueError(msg)
        self._min = min
        self._max = max

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping({"sequence": self._generator, "min": self._min, "max": self._max})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).clamp(
            min=self._min, max=self._max
        )


class CumsumSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that computes the cumulative sum
    of a generated sequence.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Cumsum, RandUniform
    >>> generator = Cumsum(RandUniform())
    >>> generator
    CumsumSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return cumsum_along_seq(
            self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        )


class DivSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator that divides one sequence by
    another one.

    ``sequence = dividend / divisor`` (a.k.a. true division)

    Args:
        dividend: The dividend sequence generator or its
            configuration.
        divisor: The divisor sequence generator or its
            configuration.
        rounding_mode: The type of rounding applied to the
            result:
                - ``None``: true division.
                - ``"trunc"``: rounds the results of the division
                    towards zero.
                - ``"floor"``: floor division.


    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Div, RandUniform, RandNormal
    >>> generator = Div(RandNormal(), RandUniform(1.0, 10.0))
    >>> generator
    DivSequenceGenerator(
      (dividend): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
      (divisor): RandUniformSequenceGenerator(low=1.0, high=10.0, feature_size=(1,))
      (rounding_mode): None
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        dividend: BaseSequenceGenerator | dict,
        divisor: BaseSequenceGenerator | dict,
        rounding_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._dividend = setup_sequence_generator(dividend)
        self._divisor = setup_sequence_generator(divisor)
        self._rounding_mode = rounding_mode

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "dividend": self._dividend,
                    "divisor": self._divisor,
                    "rounding_mode": self._rounding_mode,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._dividend.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).div(
            self._divisor.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            rounding_mode=self._rounding_mode,
        )


class ExpSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that computes the exponential of a
    batch of sequences.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Exp, RandUniform, RandNormal
    >>> generator = Exp(RandUniform())
    >>> generator
    ExpSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).exp()


class FmodSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a tensor generator that computes the element-wise
    remainder of division.

    Args:
        dividend: The sequence generator (or its
            configuration) that generates the dividend values.
        divisor: The divisor.

    Example usage:

    ```pycon

    >>> from startorch.sequence import Fmod, RandUniform
    >>> generator = Fmod(dividend=RandUniform(low=-100, high=100), divisor=10.0)
    >>> generator
    FmodSequenceGenerator(
      (dividend): RandUniformSequenceGenerator(low=-100.0, high=100.0, feature_size=(1,))
      (divisor): 10.0
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])
    >>> generator = Fmod(
    ...     dividend=RandUniform(low=-100, high=100), divisor=RandUniform(low=1, high=10)
    ... )
    >>> generator
    FmodSequenceGenerator(
      (dividend): RandUniformSequenceGenerator(low=-100.0, high=100.0, feature_size=(1,))
      (divisor): RandUniformSequenceGenerator(low=1.0, high=10.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        dividend: BaseSequenceGenerator | dict,
        divisor: BaseSequenceGenerator | dict | float,
    ) -> None:
        super().__init__()
        self._dividend = setup_sequence_generator(dividend)
        self._divisor = setup_sequence_generator(divisor) if isinstance(divisor, dict) else divisor

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"dividend": self._dividend, "divisor": self._divisor}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        seq = self._dividend.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        divisor = self._divisor
        if isinstance(divisor, BaseSequenceGenerator):
            divisor = divisor.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        seq.fmod_(divisor)
        return seq


class LogSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that computes the logarithm of a
    batch of sequences.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Log, RandUniform, RandNormal
    >>> generator = Log(RandUniform(1.0, 10.0))
    >>> generator
    LogSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=1.0, high=10.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).log()


class MulSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator that multiplies multiple
    sequences.

    ``sequence = sequence_1 * sequence_2 * ... * sequence_n``

    Args:
        sequences: The sequence generators.

    Raises:
        ValueError: if no sequence generator is provided.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Mul, RandUniform, RandNormal
    >>> generator = Mul([RandUniform(), RandNormal()])
    >>> generator
    MulSequenceGenerator(
      (0): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
      (1): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        sequences: Sequence[BaseSequenceGenerator | dict],
    ) -> None:
        super().__init__()
        if not sequences:
            msg = "No sequence generator. You need to specify at least one sequence generator"
            raise ValueError(msg)
        self._sequences = tuple(setup_sequence_generator(generator) for generator in sequences)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._sequences))}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        output = self._sequences[0].generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        for generator in self._sequences[1:]:
            output.mul_(generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng))
        return output


class MulScalarSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that multiplies a scalar value to
    a generated batch of sequences.

    Args:
        generator: The sequence generator or its
            configuration.
        value: The scalar value to multiply.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import MulScalar, RandUniform, RandNormal
    >>> generator = MulScalar(RandUniform(), 2.0)
    >>> generator
    MulScalarSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
      (value): 2.0
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        generator: BaseSequenceGenerator | dict,
        value: float,
    ) -> None:
        super().__init__(generator=generator)
        self._value = value

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"sequence": self._generator, "value": self._value}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        sequence = self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        sequence.mul_(self._value)
        return sequence


class NegSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that computes the negation of a
    generated sequence.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Neg, RandUniform, RandNormal
    >>> generator = Neg(RandUniform())
    >>> generator
    NegSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return -self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)


class SqrtSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that computes the squared root of
    a batch of sequences.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Sqrt, RandUniform, RandNormal
    >>> generator = Sqrt(RandUniform(1.0, 4.0))
    >>> generator
    SqrtSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=1.0, high=4.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).sqrt()


class SubSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator that subtracts sequences.

    ``sequence = sequence_1 - sequence_2``

    Args:
        sequence1: The first sequence generator or its
            configuration.
        sequence2: The second sequence generator or its
            configuration.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Sub, RandUniform, RandNormal
    >>> generator = Sub(RandUniform(), RandNormal())
    >>> generator
    SubSequenceGenerator(
      (sequence1): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
      (sequence2): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        sequence1: BaseSequenceGenerator | dict,
        sequence2: BaseSequenceGenerator | dict,
    ) -> None:
        super().__init__()
        self._sequence1 = setup_sequence_generator(sequence1)
        self._sequence2 = setup_sequence_generator(sequence2)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"sequence1": self._sequence1, "sequence2": self._sequence2}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._sequence1.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).sub(
            self._sequence2.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        )
