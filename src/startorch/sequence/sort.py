from __future__ import annotations

__all__ = ["SortSequenceGenerator"]


from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator
from startorch.sequence.wrapper import BaseWrapperSequenceGenerator


class SortSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that sorts a generated sequence.

    Args:
    ----
        sequence (``BaseSequenceGenerator`` or dict):
            Specifies the sequence generator or its configuration.
        descending (bool, optional): Controls the sorting order.
            If ``True``, the elements are sorted in
            descending order by value. Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import RandUniform, Sort
        >>> generator = Sort(RandUniform())
        >>> generator
        SortSequenceGenerator(
          (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        generator: BaseSequenceGenerator | dict,
        descending: bool = False,
    ) -> None:
        super().__init__(generator)
        self._descending = bool(descending)

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._generator.generate(
            seq_len=seq_len, batch_size=batch_size, rng=rng
        ).sort_along_seq(self._descending)[0]
