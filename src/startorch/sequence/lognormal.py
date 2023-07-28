from __future__ import annotations

__all__ = [
    "LogNormalSequenceGenerator",
    "RandLogNormalSequenceGenerator",
    "RandTruncLogNormalSequenceGenerator",
    "TruncLogNormalSequenceGenerator",
]


from coola.utils.format import str_indent, str_mapping
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.random import (
    log_normal,
    rand_log_normal,
    rand_trunc_log_normal,
    trunc_log_normal,
)
from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.utils.conversion import to_tuple


class LogNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    log-Normal distribution.

    Args:
    ----
        mean (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the mean
            of the underlying Normal distribution.
        std (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the standard
            deviation of the underlying Normal distribution.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import LogNormal, RandUniform
        >>> generator = LogNormal(
        ...     mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0)
        ... )
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self, mean: BaseSequenceGenerator | dict, std: BaseSequenceGenerator | dict
    ) -> None:
        super().__init__()
        self._mean = setup_sequence_generator(mean)
        self._std = setup_sequence_generator(std)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"mean": self._mean, "std": self._std}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            log_normal(
                mean=self._mean.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                std=self._std.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                generator=rng,
            )
        )


class RandLogNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    log-Normal distribution.

    Args:
    ----
        mean (float, optional): Specifies the mean of the underlying
            Normal distribution. Default: ``0.0``
        std (float, optional): Specifies the standard deviation of the
            underlying Normal distribution. Default: ``1.0``
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Raises:
    ------
        ValueError if ``std`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandLogNormal
        >>> generator = RandLogNormal(mean=0.0, std=1.0)
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        self._mean = float(mean)
        if std <= 0:
            raise ValueError(f"std has to be greater than 0 (received: {std})")
        self._std = float(std)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(mean={self._mean}, std={self._std}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_log_normal(
                size=(batch_size, seq_len) + self._feature_size,
                mean=self._mean,
                std=self._std,
                generator=rng,
            )
        )


class RandTruncLogNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator to generate cyclic sequences by
    sampling values from a truncated log-Normal distribution.

    Args:
    ----
        mean (float, optional): Specifies the mean of the
            log-Normal distribution. Default: ``0.0``
        std (float, optional): Specifies the standard deviation of
            the log-Normal distribution. Default: ``1.0``
        min_value (float, optional): Specifies the minimum value.
            Default: ``0.0``
        max_value (float, optional): Specifies the maximum value.
            Default: ``5.0``
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Raises:
    ------
        ValueError if ``std`` is not a positive number.
        ValueError if ``max_value`` is lower than ``min_value``.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandTruncLogNormal
        >>> generator = RandTruncLogNormal(mean=0.0, std=1.0, min_value=0.0, max_value=1.0)
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        min_value: float = 0.0,
        max_value: float = 5.0,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        self._mean = float(mean)
        if std <= 0:
            raise ValueError(f"std has to be greater than 0 (received: {std})")
        self._std = float(std)
        if max_value < min_value:
            raise ValueError(
                f"max_value ({max_value}) has to be greater or equal to min_value ({min_value})"
            )
        self._min_value = float(min_value)
        self._max_value = float(max_value)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(mean={self._mean}, std={self._std}, "
            f"min_value={self._min_value}, max_value={self._max_value}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_trunc_log_normal(
                size=(batch_size, seq_len) + self._feature_size,
                mean=self._mean,
                std=self._std,
                min_value=self._min_value,
                max_value=self._max_value,
                generator=rng,
            )
        )


class TruncLogNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values from a
    truncated log-Normal distribution.

    Args:
    ----
        mean (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the mean of
            the underlying Normal distribution.
        std (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the standard
            deviation of the underlying Normal distribution.
        min_value (``BaseSequenceGenerator`` or dict): Specifies a
            sequence generator (or its configuration) to generate the
            minimum value (included).
        max_value (``BaseSequenceGenerator`` or dict): Specifies a
            sequence generator (or its configuration) to generate the
            maximum value (excluded).

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import RandUniform, TruncNormal
        >>> generator = TruncNormal(
        ...     mean=RandUniform(low=-1.0, high=1.0),
        ...     std=RandUniform(low=1.0, high=2.0),
        ...     min_value=RandUniform(low=0.0, high=2.0),
        ...     max_value=RandUniform(low=5.0, high=10.0),
        ... )
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        mean: BaseSequenceGenerator | dict,
        std: BaseSequenceGenerator | dict,
        min_value: BaseSequenceGenerator | dict,
        max_value: BaseSequenceGenerator | dict,
    ) -> None:
        super().__init__()
        self._mean = setup_sequence_generator(mean)
        self._std = setup_sequence_generator(std)
        self._min_value = setup_sequence_generator(min_value)
        self._max_value = setup_sequence_generator(max_value)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "mean": self._mean,
                    "std": self._std,
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
            trunc_log_normal(
                mean=self._mean.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                std=self._std.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                min_value=self._min_value.generate(
                    seq_len=seq_len, batch_size=batch_size, rng=rng
                ).data,
                max_value=self._max_value.generate(
                    seq_len=seq_len, batch_size=batch_size, rng=rng
                ).data,
                generator=rng,
            )
        )
