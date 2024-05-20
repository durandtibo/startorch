r"""Contain the implementation of sequence generators to change the data
type of sequences/tensors."""

from __future__ import annotations

__all__ = ["FloatSequenceGenerator", "LongSequenceGenerator"]

from typing import TYPE_CHECKING

from startorch.sequence.wrapper import BaseWrapperSequenceGenerator

if TYPE_CHECKING:
    import torch


class FloatSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that converts a batch of sequences
    to float type.

    Example usage:

    ```pycon

    >>> from startorch.sequence import Float, RandInt
    >>> generator = Float(RandInt(low=0, high=10))
    >>> generator
    FloatSequenceGenerator(
      (sequence): RandIntSequenceGenerator(low=0, high=10, feature_size=())
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).float()


class LongSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that converts a batch of sequences
    to long type.

    Example usage:

    ```pycon

    >>> from startorch.sequence import Long, RandUniform
    >>> generator = Long(RandUniform(low=0, high=10))
    >>> generator
    LongSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=0.0, high=10.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).long()
