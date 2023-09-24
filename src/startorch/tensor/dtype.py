from __future__ import annotations

__all__ = ["FloatTensorGenerator", "LongTensorGenerator"]

from torch import Generator, Tensor

from startorch.tensor.wrapper import BaseWrapperTensorGenerator


class FloatTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a sequence generator that converts tensor values to
    float.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Float, RandInt
        >>> generator = Float(RandInt(low=0, high=10))
        >>> generator
        FloatTensorGenerator(
          (tensor): RandIntTensorGenerator(low=0, high=10)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._generator.generate(size=size, rng=rng).float()


class LongTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a sequence generator that converts a tensor values to
    long.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Long, RandUniform
        >>> generator = Long(RandUniform(low=0, high=10))
        >>> generator
        LongTensorGenerator(
          (tensor): RandUniformTensorGenerator(low=0.0, high=10.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._generator.generate(size=size, rng=rng).long()
