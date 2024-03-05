r"""Contain the implementation of sequence generators where the values
are sampled from a Log-Normal distribution."""

from __future__ import annotations

__all__ = [
    "LogNormalSequenceGenerator",
    "RandLogNormalSequenceGenerator",
    "RandTruncLogNormalSequenceGenerator",
    "TruncLogNormalSequenceGenerator",
]

from typing import TYPE_CHECKING

from coola.utils.format import str_indent, str_mapping

from startorch.random import (
    log_normal,
    rand_log_normal,
    rand_trunc_log_normal,
    trunc_log_normal,
)
from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.utils.conversion import to_tuple

if TYPE_CHECKING:
    import torch


class LogNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    log-Normal distribution.

    Args:
        mean: Specifies a sequence generator (or its configuration) to
            generate the mean of the underlying Normal distribution.
        std: Specifies a sequence generator (or its configuration) to
            generate the standard deviation of the underlying Normal
            distribution.

    Example usage:

    ```pycon
    >>> from startorch.sequence import LogNormal, RandUniform
    >>> generator = LogNormal(
    ...     mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0)
    ... )
    >>> generator
    LogNormalSequenceGenerator(
      (mean): RandUniformSequenceGenerator(low=-1.0, high=1.0, feature_size=(1,))
      (std): RandUniformSequenceGenerator(low=1.0, high=2.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
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
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return log_normal(
            mean=self._mean.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            std=self._std.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            generator=rng,
        )


class RandLogNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    log-Normal distribution.

    Args:
        mean: Specifies the mean of the underlying Normal distribution.
        std: Specifies the standard deviation of the underlying Normal
            distribution.
        feature_size: Specifies the feature size.

    Raises:
        ValueError: if ``std`` is not a positive number.

    Example usage:

    ```pycon
    >>> from startorch.sequence import RandLogNormal
    >>> generator = RandLogNormal(mean=0.0, std=1.0)
    >>> generator
    RandLogNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
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
            msg = f"std has to be greater than 0 (received: {std})"
            raise ValueError(msg)
        self._std = float(std)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(mean={self._mean}, std={self._std}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return rand_log_normal(
            size=(batch_size, seq_len) + self._feature_size,
            mean=self._mean,
            std=self._std,
            generator=rng,
        )


class RandTruncLogNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator to generate cyclic sequences by
    sampling values from a truncated log-Normal distribution.

    Args:
        mean: Specifies the mean of the log-Normal distribution.
        std: Specifies the standard deviation of the log-Normal
            distribution.
        min_value: Specifies the minimum value.
        max_value: Specifies the maximum value.
        feature_size: Specifies the feature size.

    Raises:
        ValueError: if ``std`` is not a positive number.
        ValueError: if ``max_value`` is lower than ``min_value``.

    Example usage:

    ```pycon
    >>> from startorch.sequence import RandTruncLogNormal
    >>> generator = RandTruncLogNormal(mean=0.0, std=1.0, min_value=0.0, max_value=1.0)
    >>> generator
    RandTruncLogNormalSequenceGenerator(mean=0.0, std=1.0, min_value=0.0, max_value=1.0, feature_size=(1,))
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
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
            msg = f"std has to be greater than 0 (received: {std})"
            raise ValueError(msg)
        self._std = float(std)
        if max_value < min_value:
            msg = f"max_value ({max_value}) has to be greater or equal to min_value ({min_value})"
            raise ValueError(msg)
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
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return rand_trunc_log_normal(
            size=(batch_size, seq_len) + self._feature_size,
            mean=self._mean,
            std=self._std,
            min_value=self._min_value,
            max_value=self._max_value,
            generator=rng,
        )


class TruncLogNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    truncated log-Normal distribution.

    Args:
        mean: Specifies a sequence
            generator (or its configuration) to generate the mean of
            the underlying Normal distribution.
        std: Specifies a sequence
            generator (or its configuration) to generate the standard
            deviation of the underlying Normal distribution.
        min_value: Specifies a
            sequence generator (or its configuration) to generate the
            minimum value (included).
        max_value: Specifies a
            sequence generator (or its configuration) to generate the
            maximum value (excluded).

    Example usage:

    ```pycon
    >>> from startorch.sequence import RandUniform, TruncLogNormal
    >>> generator = TruncLogNormal(
    ...     mean=RandUniform(low=-1.0, high=1.0),
    ...     std=RandUniform(low=1.0, high=2.0),
    ...     min_value=RandUniform(low=0.0, high=2.0),
    ...     max_value=RandUniform(low=5.0, high=10.0),
    ... )
    >>> generator
    TruncLogNormalSequenceGenerator(
      (mean): RandUniformSequenceGenerator(low=-1.0, high=1.0, feature_size=(1,))
      (std): RandUniformSequenceGenerator(low=1.0, high=2.0, feature_size=(1,))
      (min_value): RandUniformSequenceGenerator(low=0.0, high=2.0, feature_size=(1,))
      (max_value): RandUniformSequenceGenerator(low=5.0, high=10.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
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
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return trunc_log_normal(
            mean=self._mean.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            std=self._std.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            min_value=self._min_value.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            max_value=self._max_value.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            generator=rng,
        )
