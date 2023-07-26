from __future__ import annotations

__all__ = ["ConstantSequenceGenerator", "FullSequenceGenerator"]


import torch
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator
from startorch.sequence.wrapper import BaseWrapperSequenceGenerator
from startorch.utils.conversion import to_tuple


class ConstantSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator to generate a batch of sequences
    with constant values where the values for each sequence are sampled
    from another sequence generator.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import Constant, RandUniform
        >>> generator = Constant(RandUniform())
        >>> generator.generate(seq_len=6, batch_size=2)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._sequence.generate(seq_len=1, batch_size=batch_size, rng=rng).repeat_along_seq(
            seq_len
        )


class FullSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator to generate sequences filled with
    a given value.

    This sequence generator is fully deterministic and the random
    seed has no effect.

    Args:
    ----
        value (float): Specifies the value.
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import Full
        >>> generator = Full(42.0)
        >>> generator.generate(seq_len=6, batch_size=2)
        tensor([[[42.],
                 [42.],
                 [42.],
                 [42.],
                 [42.],
                 [42.]],
                [[42.],
                 [42.],
                 [42.],
                 [42.],
                 [42.],
                 [42.]]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        value: float,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        self._value = float(value)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(value={self._value}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(torch.full((batch_size, seq_len) + self._feature_size, self._value))
