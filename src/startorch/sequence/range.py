r"""Contain the implementation of sequence generators that generates
sequences from a range of values."""

from __future__ import annotations

__all__ = ["ArangeSequenceGenerator"]


import torch

from startorch.sequence.base import BaseSequenceGenerator
from startorch.utils.conversion import to_tuple


class ArangeSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence of consecutive integer
    values between ``0`` and ``seq_len-1``.

    Args:
        feature_size: Specifies the feature size.

    Example usage:

    ```pycon
    >>> from startorch.sequence import Arange
    >>> generator = Arange(feature_size=())
    >>> generator
    ArangeSequenceGenerator(feature_size=())
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5]])

    ```
    """

    def __init__(self, feature_size: tuple[int, ...] | list[int] | int = 1) -> None:
        super().__init__()
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(feature_size={self._feature_size})"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return (
            torch.arange(0, seq_len)
            .view(1, seq_len, *((1,) * len(self._feature_size)))
            .repeat(batch_size, 1, *self._feature_size)
        )
