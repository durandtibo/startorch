r"""Contain the implementation of transition matrix generator that
normalizes a transition matrix."""

from __future__ import annotations

__all__ = ["NormalizeTransitionGenerator"]


import torch
from coola.utils import str_indent, str_mapping

from startorch.transition.base import (
    BaseTransitionGenerator,
    setup_transition_generator,
)


class NormalizeTransitionGenerator(BaseTransitionGenerator):
    r"""Implement a transition matrix generator that normalizes a
    transition matrix.

    Args:
        generator: The transition generator or its configuration.
        p: The exponent value in the norm formulation.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transition import Normalize, TensorTransitionGenerator
    >>> from startorch.tensor import Full
    >>> generator = Normalize(generator=TensorTransitionGenerator(Full(1.0)))
    >>> generator
    NormalizeTransitionGenerator(
      (generator): TensorTransitionGenerator(
          (generator): FullTensorGenerator(value=1.0, dtype=None)
        )
      (p): 2.0
    )
    >>> generator.generate(n=4)
    tensor([[0.5000, 0.5000, 0.5000, 0.5000],
            [0.5000, 0.5000, 0.5000, 0.5000],
            [0.5000, 0.5000, 0.5000, 0.5000],
            [0.5000, 0.5000, 0.5000, 0.5000]])

    ```
    """

    def __init__(self, generator: BaseTransitionGenerator | dict, p: float = 2.0) -> None:
        self._generator = setup_transition_generator(generator)
        self._p = p

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"generator": self._generator, "p": self._p}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self,
        n: int,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        matrix = self._generator.generate(n=n, rng=rng)
        return torch.nn.functional.normalize(matrix, p=self._p, dim=1)
