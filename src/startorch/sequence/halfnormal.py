from __future__ import annotations

__all__ = [
    "HalfNormalSequenceGenerator",
    "RandHalfNormalSequenceGenerator",
    "RandTruncHalfNormalSequenceGenerator",
    "TruncHalfNormalSequenceGenerator",
]


from coola.utils.format import str_indent, str_mapping
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.random import (
    half_normal,
    rand_half_normal,
    rand_trunc_half_normal,
    trunc_half_normal,
)
from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.utils.conversion import to_tuple


class HalfNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    half-Normal distribution.

    Args:
    ----
        std (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the standard
            deviation.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import HalfNormal, RandUniform
        >>> generator = HalfNormal(std=RandUniform(low=1.0, high=2.0))
        >>> generator
        HalfNormalSequenceGenerator(
          (std): RandUniformSequenceGenerator(low=1.0, high=2.0, feature_size=(1,))
        )
        >>> generator.generate(seq_len=6, batch_size=2)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(self, std: BaseSequenceGenerator | dict) -> None:
        super().__init__()
        self._std = setup_sequence_generator(std)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"std": self._std}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            half_normal(
                std=self._std.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                generator=rng,
            )
        )


class RandHalfNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    half-Normal distribution.

    Args:
    ----
        std (float, optional): Specifies the std of the distribution.
            Default: ``1.0``
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Raises:
    ------
        ValueError if ``std`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandHalfNormal
        >>> generator = RandHalfNormal(std=1.0)
        >>> generator
        RandHalfNormalSequenceGenerator(std=1.0, feature_size=(1,))
        >>> generator.generate(seq_len=6, batch_size=2)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self, std: float = 1.0, feature_size: tuple[int, ...] | list[int] | int = 1
    ) -> None:
        super().__init__()
        if std <= 0:
            raise ValueError(f"std has to be greater than 0 (received: {std})")
        self._std = float(std)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(std={self._std}, feature_size={self._feature_size})"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_half_normal(
                size=(batch_size, seq_len) + self._feature_size,
                std=self._std,
                generator=rng,
            )
        )


class RandTruncHalfNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    truncated half-Normal distribution.

    Args:
    ----
        std (float, optional): Specifies the std of the distribution.
            Default: ``1.0``
        max_value (float, optional): Specifies the maximum value.
            Default: ``3.0``
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Raises:
    ------
        ValueError if ``std`` is not a positive number.
        ValueError if ``max_value`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandTruncHalfNormal
        >>> generator = RandTruncHalfNormal(std=1.0, max_value=1.0)
        >>> generator
        RandTruncHalfNormalSequenceGenerator(std=1.0, max_value=1.0, feature_size=(1,))
        >>> generator.generate(seq_len=6, batch_size=2)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        std: float = 1.0,
        max_value: float = 3.0,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        if std <= 0:
            raise ValueError(f"std has to be greater than 0 (received: {std})")
        self._std = float(std)
        if max_value <= 0:
            raise ValueError(f"max_value has to be greater than 0 (received: {max_value})")
        self._max_value = float(max_value)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(std={self._std}, max_value={self._max_value}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_trunc_half_normal(
                size=(batch_size, seq_len) + self._feature_size,
                std=self._std,
                max_value=self._max_value,
                generator=rng,
            )
        )


class TruncHalfNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    half-Normal distribution.

    Args:
    ----
        std (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the std.
        max_value (``BaseSequenceGenerator`` or dict): Specifies a
            sequence generator (or its configuration) to generate the
            maximum value (excluded).

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandUniform, TruncHalfNormal
        >>> generator = TruncHalfNormal(
        ...     std=RandUniform(low=1.0, high=2.0),
        ...     max_value=RandUniform(low=5.0, high=10.0),
        ... )
        >>> generator
        TruncHalfNormalSequenceGenerator(
          (std): RandUniformSequenceGenerator(low=1.0, high=2.0, feature_size=(1,))
          (max_value): RandUniformSequenceGenerator(low=5.0, high=10.0, feature_size=(1,))
        )
        >>> generator.generate(seq_len=6, batch_size=2)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self, std: BaseSequenceGenerator | dict, max_value: BaseSequenceGenerator | dict
    ) -> None:
        super().__init__()
        self._std = setup_sequence_generator(std)
        self._max_value = setup_sequence_generator(max_value)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"std": self._std, "max_value": self._max_value}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            trunc_half_normal(
                std=self._std.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                max_value=self._max_value.generate(
                    seq_len=seq_len, batch_size=batch_size, rng=rng
                ).data,
                generator=rng,
            )
        )
