r"""Contain the implementation of tensor generators to change the data
type of tensors."""

from __future__ import annotations

__all__ = ["FloatTensorGenerator", "LongTensorGenerator"]

from typing import TYPE_CHECKING

from startorch.tensor.wrapper import BaseWrapperTensorGenerator

if TYPE_CHECKING:
    import torch


class FloatTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implement a sequence generator that converts tensor values to
    float.

    Example usage:

    ```pycon
    >>> from startorch.tensor import Float, RandInt
    >>> generator = Float(RandInt(low=0, high=10))
    >>> generator
    FloatTensorGenerator(
      (tensor): RandIntTensorGenerator(low=0, high=10)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return self._generator.generate(size=size, rng=rng).float()


class LongTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implement a sequence generator that converts a tensor values to
    long.

    Example usage:

    ```pycon
    >>> from startorch.tensor import Long, RandUniform
    >>> generator = Long(RandUniform(low=0, high=10))
    >>> generator
    LongTensorGenerator(
      (tensor): RandUniformTensorGenerator(low=0.0, high=10.0)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return self._generator.generate(size=size, rng=rng).long()
