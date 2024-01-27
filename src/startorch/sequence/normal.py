r"""Contain the implementation of sequence generators where the values
are sampled from a Normal distribution."""

from __future__ import annotations

__all__ = [
    "NormalSequenceGenerator",
    "RandNormalSequenceGenerator",
    "RandTruncNormalSequenceGenerator",
    "TruncNormalSequenceGenerator",
]

from typing import TYPE_CHECKING

from coola.utils.format import str_indent, str_mapping
from redcat import BatchedTensorSeq

from startorch.random import normal, rand_normal, rand_trunc_normal, trunc_normal
from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.utils.conversion import to_tuple

if TYPE_CHECKING:
    from torch import Generator


class NormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    Normal distribution.

    Args:
        mean: Specifies a sequence generator (or its configuration)
            to generate the mean.
        std: Specifies a sequence generator (or its configuration) to
            generate the standard deviation.

    Example usage:

    ```pycon
    >>> from startorch.sequence import Normal, RandUniform
    >>> generator = Normal(
    ...     mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0)
    ... )
    >>> generator
    NormalSequenceGenerator(
      (mean): RandUniformSequenceGenerator(low=-1.0, high=1.0, feature_size=(1,))
      (std): RandUniformSequenceGenerator(low=1.0, high=2.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]], batch_dim=0, seq_dim=1)

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
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            normal(
                mean=self._mean.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                std=self._std.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                generator=rng,
            )
        )


class RandNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator to generate cyclic sequences by
    sampling values from a Normal distribution.

    Args:
        mean: Specifies the mean of the Normal distribution.
        std: Specifies the standard deviation of the Normal
            distribution.
        feature_size: Specifies the feature size.

    Raises:
        ValueError: if ``std`` is not a positive number.

    Example usage:

    ```pycon
    >>> from startorch.sequence import RandNormal
    >>> generator = RandNormal(mean=0.0, std=1.0)
    >>> generator
    RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]], batch_dim=0, seq_dim=1)

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
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_normal(
                size=(batch_size, seq_len) + self._feature_size,
                mean=self._mean,
                std=self._std,
                generator=rng,
            )
        )


class RandTruncNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator to generate cyclic sequences by
    sampling values from a truncated Normal distribution.

    Args:
        mean: Specifies the mean of the Normal distribution.
        std: Specifies the standard deviation of the Normal
            distribution.
        min_value: Specifies the minimum value.
        max_value: Specifies the maximum value.
        feature_size: Specifies the feature size.

    Raises:
        ValueError: if ``std`` is not a positive number.
        ValueError: if ``max_value`` is lower than ``min_value``.

    Example usage:

    ```pycon

    >>> from startorch.sequence import RandTruncNormal
    >>> generator = RandTruncNormal(mean=0.0, std=1.0, min_value=-1.0, max_value=1.0)
    >>> generator
    RandTruncNormalSequenceGenerator(mean=0.0, std=1.0, min_value=-1.0, max_value=1.0, feature_size=(1,))
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]], batch_dim=0, seq_dim=1)

    ```
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        min_value: float = -3.0,
        max_value: float = 3.0,
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
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_trunc_normal(
                size=(batch_size, seq_len) + self._feature_size,
                mean=self._mean,
                std=self._std,
                min_value=self._min_value,
                max_value=self._max_value,
                generator=rng,
            )
        )


class TruncNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    truncated Normal distribution.

    Args:
        mean: Specifies a sequence
            generator (or its configuration) to generate the mean.
        std: Specifies a sequence
            generator (or its configuration) to generate the standard
            deviation.
        min_value: Specifies a
            sequence generator (or its configuration) to generate the
            minimum value (included).
        max_value: Specifies a
            sequence generator (or its configuration) to generate the
            maximum value (excluded).

    Example usage:

    ```pycon
    >>> from startorch.sequence import RandUniform, TruncNormal
    >>> generator = TruncNormal(
    ...     mean=RandUniform(low=-1.0, high=1.0),
    ...     std=RandUniform(low=1.0, high=2.0),
    ...     min_value=RandUniform(low=-10.0, high=-5.0),
    ...     max_value=RandUniform(low=5.0, high=10.0),
    ... )
    >>> generator
    TruncNormalSequenceGenerator(
      (mean): RandUniformSequenceGenerator(low=-1.0, high=1.0, feature_size=(1,))
      (std): RandUniformSequenceGenerator(low=1.0, high=2.0, feature_size=(1,))
      (min_value): RandUniformSequenceGenerator(low=-10.0, high=-5.0, feature_size=(1,))
      (max_value): RandUniformSequenceGenerator(low=5.0, high=10.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]], batch_dim=0, seq_dim=1)

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
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            trunc_normal(
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
