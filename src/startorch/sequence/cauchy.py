from __future__ import annotations

__all__ = [
    "CauchySequenceGenerator",
    "RandCauchySequenceGenerator",
    "RandTruncCauchySequenceGenerator",
    "TruncCauchySequenceGenerator",
]


from coola.utils.format import str_indent, str_mapping
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.random import cauchy, rand_cauchy, rand_trunc_cauchy, trunc_cauchy
from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.utils.conversion import to_tuple


class CauchySequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    Cauchy distribution.

    Args:
    ----
        loc (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the location.
        scale (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the scale.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import Cauchy, RandUniform
        >>> generator = Cauchy(
        ...     loc=RandUniform(low=-1.0, high=1.0),
        ...     scale=RandUniform(low=1.0, high=2.0),
        ... )
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        loc: BaseSequenceGenerator | dict,
        scale: BaseSequenceGenerator | dict,
    ) -> None:
        super().__init__()
        self._loc = setup_sequence_generator(loc)
        self._scale = setup_sequence_generator(scale)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"loc": self._loc, "scale": self._scale}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            cauchy(
                loc=self._loc.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                scale=self._scale.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                generator=rng,
            )
        )


class RandCauchySequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    Cauchy distribution.

    Args:
    ----
        loc (float, optional): Specifies the location/median of the
            Cauchy distribution. Default: ``0.0``
        scale (float, optional): Specifies the scale of the
            distribution. Default: ``1.0``
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Raises:
    ------
        ValueError if ``scale`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandCauchy
        >>> generator = RandCauchy(loc=0.0, scale=1.0)
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        self._loc = float(loc)
        if scale <= 0:
            raise ValueError(f"scale has to be greater than 0 (received: {scale})")
        self._scale = float(scale)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(loc={self._loc}, scale={self._scale}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_cauchy(
                size=(batch_size, seq_len) + self._feature_size,
                loc=self._loc,
                scale=self._scale,
                generator=rng,
            )
        )


class RandTruncCauchySequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    truncated Cauchy distribution.

    Args:
    ----
        loc (float, optional): Specifies the location/median of the
            Cauchy distribution. Default: ``0.0``
        scale (float, optional): Specifies the scale of the
            distribution. Default: ``1.0``
        min_value (float, optional): Specifies the minimum value.
            (included). Default: ``-2.0``
        max_value (float, optional): Specifies the maximum value
            (excluded). Default: ``2.0``
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Raises:
    ------
        ValueError if ``std`` is not a positive number.
        ValueError if ``max_value`` is lower than ``min_value``.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandTruncCauchy
        >>> generator = RandTruncCauchy(loc=0.0, scale=1.0, min_value=-5.0, max_value=5.0)
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        min_value: float = -2.0,
        max_value: float = 2.0,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        self._loc = float(loc)
        if scale <= 0:
            raise ValueError(f"scale has to be greater than 0 (received: {scale})")
        self._scale = float(scale)
        if max_value < min_value:
            raise ValueError(
                f"max_value ({max_value}) has to be greater or equal to min_value ({min_value})"
            )
        self._min_value = float(min_value)
        self._max_value = float(max_value)

        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(loc={self._loc}, scale={self._scale}, "
            f"min_value={self._min_value}, max_value={self._max_value}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_trunc_cauchy(
                size=(batch_size, seq_len) + self._feature_size,
                loc=self._loc,
                scale=self._scale,
                min_value=self._min_value,
                max_value=self._max_value,
                generator=rng,
            )
        )


class TruncCauchySequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    Cauchy distribution.

    Args:
    ----
        loc (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the location.
        scale (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the scale.
        min_value (``BaseSequenceGenerator`` or dict): Specifies a
            sequence generator (or its configuration) to generate the
            minimum value (included).
        max_value (``BaseSequenceGenerator`` or dict): Specifies a
            sequence generator (or its configuration) to generate the
            maximum value (excluded).

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandUniform, TruncCauchy
        >>> generator = TruncCauchy(
        ...     loc=RandUniform(low=-1.0, high=1.0),
        ...     scale=RandUniform(low=1.0, high=2.0),
        ...     min_value=RandUniform(low=-10.0, high=-5.0),
        ...     max_value=RandUniform(low=5.0, high=10.0),
        ... )
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        loc: BaseSequenceGenerator | dict,
        scale: BaseSequenceGenerator | dict,
        min_value: BaseSequenceGenerator | dict,
        max_value: BaseSequenceGenerator | dict,
    ) -> None:
        super().__init__()
        self._loc = setup_sequence_generator(loc)
        self._scale = setup_sequence_generator(scale)
        self._min_value = setup_sequence_generator(min_value)
        self._max_value = setup_sequence_generator(max_value)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "loc": self._loc,
                    "scale": self._scale,
                    "min_value": self._min_value,
                    "max_value": self._max_value,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            trunc_cauchy(
                loc=self._loc.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                scale=self._scale.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                min_value=self._min_value.generate(
                    seq_len=seq_len, batch_size=batch_size, rng=rng
                ).data,
                max_value=self._max_value.generate(
                    seq_len=seq_len, batch_size=batch_size, rng=rng
                ).data,
                generator=rng,
            )
        )
