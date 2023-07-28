from __future__ import annotations

__all__ = [
    "HalfCauchySequenceGenerator",
    "RandHalfCauchySequenceGenerator",
    "RandTruncHalfCauchySequenceGenerator",
    "TruncHalfCauchySequenceGenerator",
]


from coola.utils.format import str_indent, str_mapping
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.random import (
    half_cauchy,
    rand_half_cauchy,
    rand_trunc_half_cauchy,
    trunc_half_cauchy,
)
from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.utils.conversion import to_tuple


class HalfCauchySequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    half-Cauchy distribution.

    Args:
    ----
        scale (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the scale.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import HalfCauchy, RandUniform
        >>> generator = HalfCauchy(scale=RandUniform(low=1.0, high=2.0))
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(self, scale: BaseSequenceGenerator | dict) -> None:
        super().__init__()
        self._scale = setup_sequence_generator(scale)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"scale": self._scale}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            half_cauchy(
                scale=self._scale.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                generator=rng,
            )
        )


class RandHalfCauchySequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    half-Cauchy distribution.

    Args:
    ----
        scale (float, optional): Specifies the scale of the
            distribution. Default: ``1.0``
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Raises:
    ------
        ValueError if ``scale`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandHalfCauchy
        >>> generator = RandHalfCauchy(scale=1.0)
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        scale: float = 1.0,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale has to be greater than 0 (received: {scale})")
        self._scale = float(scale)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(scale={self._scale}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_half_cauchy(
                size=(batch_size, seq_len) + self._feature_size,
                scale=self._scale,
                generator=rng,
            )
        )


class RandTruncHalfCauchySequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    truncated half-Cauchy distribution.

    Args:
    ----
        scale (float, optional): Specifies the scale of the
            distribution. Default: ``1.0``
        max_value (float, optional): Specifies the maximum value.
            Default: ``4.0``
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Raises:
    ------
        ValueError if ``scale`` is not a positive number.
        ValueError if ``max_value`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandTruncHalfCauchy
        >>> generator = RandTruncHalfCauchy(scale=1.0, max_value=5.0)
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        scale: float = 1.0,
        max_value: float = 4.0,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale has to be greater than 0 (received: {scale})")
        self._scale = float(scale)
        if max_value <= 0:
            raise ValueError(f"max_value has to be greater than 0 (received: {max_value})")
        self._max_value = float(max_value)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(scale={self._scale}, max_value={self._max_value}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_trunc_half_cauchy(
                size=(batch_size, seq_len) + self._feature_size,
                scale=self._scale,
                max_value=self._max_value,
                generator=rng,
            )
        )


class TruncHalfCauchySequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    half-Cauchy distribution.

    Args:
    ----
        scale (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the scale.
        max_value (``BaseSequenceGenerator`` or dict): Specifies a
            sequence generator (or its configuration) to generate the
            maximum value (excluded).

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandUniform, TruncHalfCauchy
        >>> generator = TruncHalfCauchy(
        ...     scale=RandUniform(low=1.0, high=2.0),
        ...     max_value=RandUniform(low=5.0, high=10.0),
        ... )
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self, scale: BaseSequenceGenerator | dict, max_value: BaseSequenceGenerator | dict
    ) -> None:
        super().__init__()
        self._scale = setup_sequence_generator(scale)
        self._max_value = setup_sequence_generator(max_value)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"scale": self._scale, "max_value": self._max_value}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            trunc_half_cauchy(
                scale=self._scale.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                max_value=self._max_value.generate(
                    seq_len=seq_len, batch_size=batch_size, rng=rng
                ).data,
                generator=rng,
            )
        )
