from __future__ import annotations

__all__ = ["FloatSequenceGenerator", "LongSequenceGenerator"]

from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.wrapper import BaseWrapperSequenceGenerator


class FloatSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that converts a batch of
    sequences to float type.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import Float, RandInt
        >>> generator = Float(RandInt(low=0, high=10))
        >>> generator
        FloatSequenceGenerator(
          (sequence): RandIntSequenceGenerator(low=0, high=10, feature_size=())
        )
        >>> generator.generate(seq_len=6, batch_size=2)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).float()


class LongSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that converts a batch of
    sequences to long type.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import Long, RandUniform
        >>> generator = Long(RandUniform(low=0, high=10))
        >>> generator
        LongSequenceGenerator(
          (sequence): RandUniformSequenceGenerator(low=0.0, high=10.0, feature_size=(1,))
        )
        >>> generator.generate(seq_len=6, batch_size=2)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).long()
