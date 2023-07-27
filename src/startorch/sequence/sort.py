from __future__ import annotations

__all__ = ["SortSequenceGenerator"]


from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator
from startorch.sequence.wrapper import BaseWrapperSequenceGenerator


class SortSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that sorts a generated sequence.

    Args:
        sequence (``BaseSequenceGenerator`` or dict):
            Specifies the sequence generator or its configuration.
        descending (bool, optional): Controls the sorting order.
            If ``True``, the elements are sorted in
            descending order by value. Default: ``False``
    """

    def __init__(
        self,
        sequence: BaseSequenceGenerator | dict,
        descending: bool = False,
    ) -> None:
        super().__init__(sequence)
        self._descending = bool(descending)

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._sequence.generate(
            seq_len=seq_len, batch_size=batch_size, rng=rng
        ).sort_along_seq(self._descending)[0]
