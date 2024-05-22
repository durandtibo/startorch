r"""Contain the implementation of a tensor generator that select a
tensor generator at each batch."""

from __future__ import annotations

__all__ = ["MultinomialChoiceTensorGenerator"]

from typing import TYPE_CHECKING

import torch
from coola.utils.format import str_indent

from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator
from startorch.utils.format import str_weighted_modules
from startorch.utils.weight import prepare_weighted_generators

if TYPE_CHECKING:
    from collections.abc import Sequence


class MultinomialChoiceTensorGenerator(BaseTensorGenerator):
    r"""Implement a tensor generator that select a tensor generator at
    each batch.

    This tensor generator is used to generate tensors with different
    generation processes. The user can specify a list of tensor
    generators with an associated weight. The weight is used to sample
    the tensor generator with a multinomial distribution. Higher
    weight means that the tensor generator has a higher probability
    to be selected at each batch. Each dictionary in the
    ``generators`` input should have the following items:

        - a key ``'generator'`` which indicates the tensor generator
            or its configuration.
        - an optional key ``'weight'`` with a float value which
            indicates the weight of the tensor generator.
            If this key is absent, the weight is set to ``1.0``.

    Args:
        generators: The tensor generators and their weights.
            See above to learn about the expected format.

    Example usage:

    ```pycon

    >>> from startorch.tensor import MultinomialChoice, RandUniform, RandNormal
    >>> generator = MultinomialChoice(
    ...     (
    ...         {"weight": 2.0, "generator": RandUniform()},
    ...         {"weight": 1.0, "generator": RandNormal()},
    ...     )
    ... )
    >>> generator
    MultinomialChoiceTensorGenerator(
      (0) [weight=2.0] RandUniformTensorGenerator(low=0.0, high=1.0)
      (1) [weight=1.0] RandNormalTensorGenerator(mean=0.0, std=1.0)
    )
    >>> generator.generate(size=(4, 12))
    tensor([[...]])

    ```
    """

    def __init__(self, generators: Sequence[dict[str, BaseTensorGenerator | dict]]) -> None:
        super().__init__()
        generators, weights = prepare_weighted_generators(generators)
        self._generators = tuple(setup_tensor_generator(generator) for generator in generators)
        self._weights = torch.as_tensor(weights, dtype=torch.float)

    def __repr__(self) -> str:
        args = str_indent(str_weighted_modules(modules=self._generators, weights=self._weights))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        index = torch.multinomial(self._weights, num_samples=1, generator=rng).item()
        return self._generators[index].generate(size=size, rng=rng)
