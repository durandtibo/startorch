from __future__ import annotations

__all__ = ["TransformTensorGenerator"]

from typing import TYPE_CHECKING

from coola.utils.format import str_indent, str_mapping

from startorch.tensor.transformer.base import (
    BaseTensorTransformer,
    setup_tensor_transformer,
)
from startorch.tensor.wrapper import BaseWrapperTensorGenerator

if TYPE_CHECKING:
    import torch

    from startorch.tensor.base import BaseTensorGenerator


class TransformTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implement a tensor generator that generates a tensor and then
    transform it.

    Args:
        generator: The tensor generator or its configuration.
        transformer: The tensor transformer or its configuration.

    Example usage:

    ```pycon

    >>> from startorch.tensor import TransformTensorGenerator, Full
    >>> from startorch.tensor.transformer import Abs
    >>> generator = TransformTensorGenerator(generator=Full(-1), transformer=Abs())
    >>> generator
    TransformTensorGenerator(
      (generator): FullTensorGenerator(value=-1, dtype=None)
      (transformer): AbsTensorTransformer()
    )
    >>> generator.generate(size=(2, 6))
    tensor([[1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]])

    ```
    """

    def __init__(
        self,
        generator: BaseTensorGenerator | dict,
        transformer: BaseTensorTransformer | dict,
    ) -> None:
        super().__init__(generator=generator)
        self._transformer = setup_tensor_transformer(transformer)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping({"generator": self._generator, "transformer": self._transformer})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        tensor = self._generator.generate(size=size, rng=rng)
        return self._transformer.transform(tensor=tensor, rng=rng)
