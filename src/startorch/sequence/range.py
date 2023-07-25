from __future__ import annotations

__all__ = ["ArangeSequenceGenerator"]


import torch
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator
from startorch.utils.conversion import to_tuple


class ArangeSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence of consecutive integer
    values between ``0`` and ``seq_len-1``.

    Args:
    ----
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import Arange
        >>> generator = Arange(feature_size=())
        >>> generator.generate(seq_len=6, batch_size=2)
        tensor([[0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5]], batch_dim=0, seq_dim=1)
    """

    def __init__(self, feature_size: tuple[int, ...] | list[int] | int = 1) -> None:
        super().__init__()
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(feature_size={self._feature_size})"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            torch.arange(0, seq_len)
            .view(1, seq_len, *((1,) * len(self._feature_size)))
            .repeat(batch_size, 1, *self._feature_size)
        )
