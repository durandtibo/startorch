r"""Contain the implementation of transition matrix generator that uses
a tensor generator."""

from __future__ import annotations

__all__ = ["MaskTransitionGenerator"]

from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.transition.base import (
    BaseTransitionGenerator,
    setup_transition_generator,
)
from startorch.utils.mask import mask_square_matrix

if TYPE_CHECKING:
    import torch


class MaskTransitionGenerator(BaseTransitionGenerator):
    r"""Implement a transition matrix generator that uses a tensor
    generator.

    Args:
        generator: The transition generator or its configuration.
        num_mask: The number of values to mask on each row of the
            transition matrix.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transition import MaskTransitionGenerator, TensorTransitionGenerator
    >>> from startorch.tensor import Full
    >>> generator = MaskTransitionGenerator(
    ...     generator=TensorTransitionGenerator(Full(1.0)), num_mask=4
    ... )
    >>> generator
    MaskTransitionGenerator(
      (generator): TensorTransitionGenerator(
          (generator): FullTensorGenerator(value=1.0, dtype=None)
        )
      (num_mask): 4
    )
    >>> generator.generate(n=6)
    tensor([[...]])

    ```
    """

    def __init__(self, generator: BaseTransitionGenerator | dict, num_mask: int) -> None:
        self._generator = setup_transition_generator(generator)
        self._num_mask = num_mask

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"generator": self._generator, "num_mask": self._num_mask}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self,
        n: int,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        return mask_square_matrix(self._generator.generate(n=n, rng=rng), n=self._num_mask, rng=rng)
