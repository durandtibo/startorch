r"""Contain a sequence generator to "generate" the input data."""

from __future__ import annotations

__all__ = ["VanillaSequenceGenerator"]

from typing import TYPE_CHECKING

from batchtensor.tensor import slice_along_batch, slice_along_seq

from startorch.sequence.base import BaseSequenceGenerator

if TYPE_CHECKING:
    import torch


class VanillaSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator to "generate" the input data.

    Args:
        data: The sequence data to generate.

    Sequence usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import VanillaSequenceGenerator
    >>> generator = VanillaSequenceGenerator(data=torch.arange(40).view(4, 10))
    >>> generator
    VanillaSequenceGenerator(batch_size=4, seq_len=10)
    >>> generator.generate(batch_size=3, seq_len=6)
    tensor([[ 0,  1,  2,  3,  4,  5],
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25]])

    ```
    """

    def __init__(self, data: torch.Tensor) -> None:
        super().__init__()
        self._data = data
        self._batch_size = data.shape[0]
        self._seq_len = data.shape[1]

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(batch_size={self._batch_size:,}, seq_len={self._seq_len:,})"

    def generate(
        self,
        seq_len: int,
        batch_size: int = 1,
        rng: torch.Generator | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        if batch_size > self._batch_size:
            msg = (
                f"Incorrect batch_size: {batch_size:,}. "
                f"batch_size cannot be greater than {self._batch_size:,}"
            )
            raise RuntimeError(msg)
        if seq_len > self._seq_len:
            msg = (
                f"Incorrect seq_len: {seq_len:,}. seq_len cannot be greater than {self._seq_len:,}"
            )
            raise RuntimeError(msg)
        data = slice_along_batch(self._data, stop=batch_size)
        return slice_along_seq(data, stop=seq_len)
