r"""Contain the implementation of transition matrix generator that uses
a tensor generator."""

from __future__ import annotations

__all__ = ["TensorTransitionGenerator"]

from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.tensor import BaseTensorGenerator, setup_tensor_generator
from startorch.transition import BaseTransitionGenerator

if TYPE_CHECKING:
    import torch


class TensorTransitionGenerator(BaseTransitionGenerator):
    r"""Implement a transition matrix generator that uses a tensor
    generator.

    Args:
        generator: The tensor generator or its configuration.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transition import TensorTransitionGenerator
    >>> from startorch.tensor import RandUniform
    >>> generator = TensorTransitionGenerator(generator=RandUniform())
    >>> generator
    TensorTransitionGenerator(
      (generator): RandUniformTensorGenerator(low=0.0, high=1.0)
    )
    >>> generator.generate(n=6)
    tensor([[...]])

    ```
    """

    def __init__(self, generator: BaseTensorGenerator | dict) -> None:
        self._generator = setup_tensor_generator(generator)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"generator": self._generator}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self,
        n: int,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self._generator.generate(size=(n, n), rng=rng)
