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

from collections.abc import Sequence

from coola.utils.format import str_indent, str_mapping, str_sequence
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.sequence.wrapper import BaseWrapperSequenceGenerator


class AbsSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that computes the absolute value
    of a generated sequence."""

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).abs()


class AddSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator that adds multiple sequences.

    ``sequence = sequence_1 + sequence_2 + ... + sequence_n``

    Args:
        sequences (``Sequence``): Specifies the sequence generators.
    """

    def __init__(
        self,
        sequences: Sequence[BaseSequenceGenerator | dict],
    ) -> None:
        super().__init__()
        if not sequences:
            raise ValueError(
                "No sequence generator. You need to specify at least one sequence generator"
            )
        self._sequences = tuple(setup_sequence_generator(generator) for generator in sequences)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._sequences))}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        output = self._sequences[0].generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        for sequence in self._sequences[1:]:
            output.add_(sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng))
        return output


class AddScalarSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that adds a scalar value to a
    generated batch of sequences.

    Args:
        sequence (``BaseSequenceGenerator`` or dict):
            Specifies the sequence generator or its configuration.
        value (int or float): Specifies the scalar value to add.
    """

    def __init__(
        self,
        sequence: BaseSequenceGenerator | dict,
        value: int | float,
    ) -> None:
        super().__init__(generator=sequence)
        self._value = value

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"sequence": self._generator, "value": self._value}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        sequence = self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        sequence.add_(self._value)
        return sequence


class ClampSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator to generate a batch of sequences
    where the values are clamped.

    Note: ``min_value`` and ``max_value`` cannot be both ``None``.

    Args:
        sequence (``BaseSequenceGenerator`` or dict):
            Specifies the sequence generator or its configuration.
        min (int, float or ``None``): Specifies the lower bound.
            If ``min_value`` is ``None``, there is no lower bound.
        max (int, float or ``None``): Specifies the upper bound.
            If ``max_value`` is ``None``, there is no upper bound.
    """

    def __init__(
        self,
        sequence: BaseSequenceGenerator | dict,
        min: int | float | None,  # noqa: A002
        max: int | float | None,  # noqa: A002
    ) -> None:
        super().__init__(generator=sequence)
        if min is None and max is None:
            raise ValueError("`min` and `max` cannot be both None")
        self._min = min
        self._max = max

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping({"sequence": self._generator, "min": self._min, "max": self._max})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).clamp(
            min=self._min, max=self._max
        )


class CumsumSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that computes the cumulative sum
    of a generated sequence."""

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._generator.generate(
            seq_len=seq_len, batch_size=batch_size, rng=rng
        ).cumsum_along_seq()


class DivSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator that divides one sequence by
    another one.

    ``sequence = dividend / divisor`` (a.k.a. true division)

    Args:
        dividend (``Sequence``): (``BaseSequenceGenerator`` or dict):
            Specifies the dividend sequence generator or its
            configuration.
        divisor (``Sequence``): (``BaseSequenceGenerator`` or dict):
            Specifies the divisor sequence generator or its
            configuration.
        rounding_mode (str or ``None``, optional): Specifies the
            type of rounding applied to the result.
            - ``None``: true division.
            - ``"trunc"``: rounds the results of the division
                towards zero.
            - ``"floor"``: floor division.
            Default: ``None``
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
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._dividend.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).div(
            self._divisor.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            rounding_mode=self._rounding_mode,
        )


class ExpSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that computes the exponential of
    a batch of sequences."""

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).exp()


class FmodSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a tensor generator that computes the element-wise
    remainder of division.

    Args:
        dividend (``BaseSequenceGenerator`` or dict):
            Specifies the sequence generator (or its configuration) that
            generates the dividend values.
        divisor (int or float): Specifies the divisor.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import Fmod, RandUniform
        >>> generator = Fmod(dividend=RandUniform(low=-100, high=100), divisor=10.0)
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
        >>> generator = Fmod(
        ...     dividend=RandUniform(low=-100, high=100), divisor=RandUniform(low=1, high=10)
        ... )
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        dividend: BaseSequenceGenerator | dict,
        divisor: BaseSequenceGenerator | dict | int | float,
    ) -> None:
        super().__init__()
        self._dividend = setup_sequence_generator(dividend)
        self._divisor = setup_sequence_generator(divisor) if isinstance(divisor, dict) else divisor

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"dividend": self._dividend, "divisor": self._divisor}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        seq = self._dividend.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        divisor = self._divisor
        if isinstance(divisor, BaseSequenceGenerator):
            divisor = divisor.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        seq.fmod_(divisor)
        return seq


class LogSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that computes the logarithm of a
    batch of sequences."""

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).log()


class MulSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator that multiplies multiple
    sequences.

    ``sequence = sequence_1 * sequence_2 * ... * sequence_n``

    Args:
        sequences (``Sequence``): Specifies the sequence generators.
    """

    def __init__(
        self,
        sequences: Sequence[BaseSequenceGenerator | dict],
    ) -> None:
        super().__init__()
        if not sequences:
            raise ValueError(
                "No sequence generator. You need to specify at least one sequence generator"
            )
        self._sequences = tuple(setup_sequence_generator(generator) for generator in sequences)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._sequences))}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        output = self._sequences[0].generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        for generator in self._sequences[1:]:
            output.mul_(generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng))
        return output


class MulScalarSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that multiplies a scalar value to
    a generated batch of sequences.

    Args:
        sequence (``BaseSequenceGenerator`` or dict):
            Specifies the sequence generator or its configuration.
        value (int or float): Specifies the scalar value to multiply.
    """

    def __init__(
        self,
        sequence: BaseSequenceGenerator | dict,
        value: int | float,
    ) -> None:
        super().__init__(generator=sequence)
        self._value = value

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"sequence": self._generator, "value": self._value}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        sequence = self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        sequence.mul_(self._value)
        return sequence


class NegSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that computes the negation of a
    generated sequence."""

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return -self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)


class SqrtSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that computes the squared root
    of a batch of sequences."""

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).sqrt()


class SubSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator that subtracts sequences.

    ``sequence = sequence_1 - sequence_2``

    Args:
        sequence1 (``BaseSequenceGenerator`` or dict):
            Specifies the first sequence generator or its
            configuration.
        sequence2 (``BaseSequenceGenerator`` or dict):
            Specifies the second sequence generator or its
            configuration.
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
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._sequence1.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).sub(
            self._sequence2.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        )