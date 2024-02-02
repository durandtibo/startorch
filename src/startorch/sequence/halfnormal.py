r"""Contain the implementation of sequence generators where the values
are sampled from a half-Normal distribution."""

from __future__ import annotations

__all__ = [
    "HalfNormalSequenceGenerator",
    "RandHalfNormalSequenceGenerator",
    "RandTruncHalfNormalSequenceGenerator",
    "TruncHalfNormalSequenceGenerator",
]

from typing import TYPE_CHECKING

from coola.utils.format import str_indent, str_mapping

from startorch.random import (
    half_normal,
    rand_half_normal,
    rand_trunc_half_normal,
    trunc_half_normal,
)
from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.utils.conversion import to_tuple

if TYPE_CHECKING:
    import torch


class HalfNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    half-Normal distribution.

    Args:
        std: Specifies a sequence generator (or its configuration)
            to generate the standard deviation.

    Example usage:

    ```pycon
    >>> from startorch.sequence import HalfNormal, RandUniform
    >>> generator = HalfNormal(std=RandUniform(low=1.0, high=2.0))
    >>> generator
    HalfNormalSequenceGenerator(
      (std): RandUniformSequenceGenerator(low=1.0, high=2.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
    """

    def __init__(self, std: BaseSequenceGenerator | dict) -> None:
        super().__init__()
        self._std = setup_sequence_generator(std)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"std": self._std}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return half_normal(
            std=self._std.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            generator=rng,
        )


class RandHalfNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    half-Normal distribution.

    Args:
        std: Specifies the std of the distribution.
        feature_size: Specifies the feature size.

    Raises:
        ValueError: if ``std`` is not a positive number.

    Example usage:

    ```pycon
    >>> from startorch.sequence import RandHalfNormal
    >>> generator = RandHalfNormal(std=1.0)
    >>> generator
    RandHalfNormalSequenceGenerator(std=1.0, feature_size=(1,))
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
    """

    def __init__(
        self, std: float = 1.0, feature_size: tuple[int, ...] | list[int] | int = 1
    ) -> None:
        super().__init__()
        if std <= 0:
            msg = f"std has to be greater than 0 (received: {std})"
            raise ValueError(msg)
        self._std = float(std)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(std={self._std}, feature_size={self._feature_size})"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return rand_half_normal(
            size=(batch_size, seq_len) + self._feature_size,
            std=self._std,
            generator=rng,
        )


class RandTruncHalfNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    truncated half-Normal distribution.

    Args:
        std: Specifies the std of the distribution.
        max_value: Specifies the maximum value.
        feature_size: Specifies the feature size.

    Raises:
        ValueError: if ``std`` is not a positive number.
        ValueError: if ``max_value`` is not a positive number.

    Example usage:

    ```pycon
    >>> from startorch.sequence import RandTruncHalfNormal
    >>> generator = RandTruncHalfNormal(std=1.0, max_value=1.0)
    >>> generator
    RandTruncHalfNormalSequenceGenerator(std=1.0, max_value=1.0, feature_size=(1,))
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        std: float = 1.0,
        max_value: float = 3.0,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        if std <= 0:
            msg = f"std has to be greater than 0 (received: {std})"
            raise ValueError(msg)
        self._std = float(std)
        if max_value <= 0:
            msg = f"max_value has to be greater than 0 (received: {max_value})"
            raise ValueError(msg)
        self._max_value = float(max_value)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(std={self._std}, max_value={self._max_value}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return rand_trunc_half_normal(
            size=(batch_size, seq_len) + self._feature_size,
            std=self._std,
            max_value=self._max_value,
            generator=rng,
        )


class TruncHalfNormalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    half-Normal distribution.

    Args:
        std: Specifies a sequence generator (or its configuration) to
            generate the std.
        max_value: Specifies a sequence generator (or its
            configuration) to generate the maximum value (excluded).

    Example usage:

    ```pycon
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
    tensor([[...]])

    ```
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
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return trunc_half_normal(
            std=self._std.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            max_value=self._max_value.generate(seq_len=seq_len, batch_size=batch_size, rng=rng),
            generator=rng,
        )
