from __future__ import annotations

__all__ = ["MultinomialChoiceTensorGenerator"]

from collections.abc import Sequence

import torch
from coola.utils.format import str_indent
from torch import Generator, Tensor

from startorch.tensor.base import BaseTensorGenerator
from startorch.tensor.preparation import prepare_weighted_generators
from startorch.utils.format import str_pretty_weighted_modules


class MultinomialChoiceTensorGenerator(BaseTensorGenerator):
    r"""Implements a tensor generator that select a tensor generator at
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
        generators (tuple or list): Specifies the tensor
            generators and their weights. See above to learn
            about the expected format.

    Sequence usage:

    .. code-block:: pycon

        >>> from startorch.tensor import MultinomialChoice, RandUniform, RandNormal
        >>> generator = MultinomialChoice(
        ...     (
        ...         {"weight": 2.0, "generator": RandUniform()},
        ...         {"weight": 1.0, "generator": RandNormal()},
        ...     )
        ... )
        >>> generator.generate(size=(4, 12))  # doctest:+ELLIPSIS
        tensor([[...]])
    """

    def __init__(self, generators: Sequence[dict[str, BaseTensorGenerator | dict]]) -> None:
        super().__init__()
        self._generators, self._weights = prepare_weighted_generators(generators)

    def __repr__(self) -> str:
        generators_str = str_indent(
            str_pretty_weighted_modules(modules=self._generators, weights=self._weights),
            num_spaces=4,
        )
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  generators:\n    {generators_str}\n"
            f"  random_seed={self.get_random_seed()},\n"
            f"  auto_set_random_seed={self._auto_set_random_seed},\n"
            ")"
        )

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        index = torch.multinomial(self._weights, num_samples=1, generator=rng).item()
        return self._generators[index].generate(size)
