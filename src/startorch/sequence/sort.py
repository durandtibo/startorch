r"""Contain the implementation of sequence generator that generates a
sequence by sorting a generated sequence."""

from __future__ import annotations

__all__ = ["SortSequenceGenerator"]

from typing import TYPE_CHECKING

from batchtensor.tensor import sort_along_seq

from startorch.sequence.wrapper import BaseWrapperSequenceGenerator

if TYPE_CHECKING:
    import torch

    from startorch.sequence.base import BaseSequenceGenerator


class SortSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that sorts a generated sequence.

    Args:
        generator: The sequence generator or its configuration.
        descending: Control the sorting order. If ``True``,
            the elements are sorted in descending order by value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import RandUniform, Sort
    >>> generator = Sort(RandUniform())
    >>> generator
    SortSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        generator: BaseSequenceGenerator | dict,
        descending: bool = False,
    ) -> None:
        super().__init__(generator)
        self._descending = bool(descending)

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        data = self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        return sort_along_seq(data, self._descending)[0]
