r"""Contain the implementation of example generator that generates
sequences from a time series generator."""

from __future__ import annotations

__all__ = ["TensorExampleGenerator"]

from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.example.base import BaseExampleGenerator
from startorch.tensor.base import setup_tensor_generator

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    import torch

    from startorch.tensor import BaseTensorGenerator


class TensorExampleGenerator(BaseExampleGenerator):
    r"""Implement an example generator to generate time series.

    Args:
        generators: The tensor generators or their configurations.
        size: The output tensor shape excepts the first dimension
            which is set to ``batch_size``.

    Example usage:

    ```pycon

    >>> from startorch.example import TensorExampleGenerator
    >>> from startorch.tensor import RandInt, RandUniform
    >>> generator = TensorExampleGenerator(
    ...     generators={"value": RandUniform(), "time": RandUniform()},
    ...     size=(6,),
    ... )
    >>> generator
    TensorExampleGenerator(
      (value): RandUniformTensorGenerator(low=0.0, high=1.0)
      (time): RandUniformTensorGenerator(low=0.0, high=1.0)
      (size): (6,)
    )
    >>> generator.generate(batch_size=10)
    {'value': tensor([[...]]), 'time': tensor([[...]])}

    ```
    """

    def __init__(
        self,
        generators: Mapping[str, BaseTensorGenerator | dict],
        size: Sequence[int],
    ) -> None:
        super().__init__()
        self._generators = {
            key: setup_tensor_generator(generator) for key, generator in generators.items()
        }
        self._size = tuple(size)

    def __repr__(self) -> str:
        args = str_indent(str_mapping(self._generators | {"size": self._size}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        size = (batch_size, *self._size)
        return {
            key: generator.generate(size=size, rng=rng)
            for key, generator in self._generators.items()
        }
