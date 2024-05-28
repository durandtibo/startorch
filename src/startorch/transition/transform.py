r"""Contain the implementation of transition matrix generator that
normalizes a transition matrix."""

from __future__ import annotations

__all__ = ["TransformTransitionGenerator"]


from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.tensor.transformer import BaseTensorTransformer, setup_tensor_transformer
from startorch.transition.base import (
    BaseTransitionGenerator,
    setup_transition_generator,
)

if TYPE_CHECKING:
    import torch


class TransformTransitionGenerator(BaseTransitionGenerator):
    r"""Implement a transition matrix generator that transform a
    transition matrix.

    Args:
        generator: The transition generator or its configuration.
        transformer: The tensor transformer or its configuration.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transition import Transform, Diagonal
    >>> from startorch.tensor.transformer import Clamp
    >>> generator = Transform(generator=Diagonal(), transformer=Clamp(min=0.0, max=0.5))
    >>> generator
    TransformTransitionGenerator(
      (generator): DiagonalTransitionGenerator()
      (transformer): ClampTensorTransformer(min=0.0, max=0.5)
    )
    >>> generator.generate(n=4)
    tensor([[0.5000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.5000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.5000]])

    ```
    """

    def __init__(
        self,
        generator: BaseTransitionGenerator | dict,
        transformer: BaseTensorTransformer | dict,
    ) -> None:
        self._generator = setup_transition_generator(generator)
        self._transformer = setup_tensor_transformer(transformer)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping({"generator": self._generator, "transformer": self._transformer})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self,
        n: int,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        matrix = self._generator.generate(n=n, rng=rng)
        return self._transformer.transform(matrix, rng=rng)
