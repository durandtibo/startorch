r"""Contain the implementations of sequences generators where the values
are constant."""

from __future__ import annotations

__all__ = ["ConstantSequenceGenerator", "FullSequenceGenerator"]


import torch
from batchtensor.tensor import repeat_along_seq

from startorch.sequence.base import BaseSequenceGenerator
from startorch.sequence.wrapper import BaseWrapperSequenceGenerator
from startorch.utils.conversion import to_tuple


class ConstantSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator to generate a batch of sequences
    with constant values where the values for each sequence are sampled
    from another sequence generator.

    Example usage:

    ```pycon

    >>> from startorch.sequence import Constant, RandUniform
    >>> generator = Constant(RandUniform())
    >>> generator
    ConstantSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[...]])

    ```
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        data = self._generator.generate(seq_len=1, batch_size=batch_size, rng=rng)
        return repeat_along_seq(data, seq_len)


class FullSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator to generate sequences filled with
    a given value.

    This sequence generator is fully deterministic and the random
    seed has no effect.

    Args:
        value: The value.
        feature_size: The feature size.

    Example usage:

    ```pycon

    >>> from startorch.sequence import Full
    >>> generator = Full(42.0)
    >>> generator
    FullSequenceGenerator(value=42.0, feature_size=(1,))
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
             [42.]]])
    >>> generator = Full(42, feature_size=())
    >>> generator
    FullSequenceGenerator(value=42, feature_size=())
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[42, 42, 42, 42, 42, 42],
            [42, 42, 42, 42, 42, 42]])

    ```
    """

    def __init__(
        self,
        value: float,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        self._value = value
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(value={self._value}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self,
        seq_len: int,
        batch_size: int = 1,
        rng: torch.Generator | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return torch.full((batch_size, seq_len, *self._feature_size), self._value)
