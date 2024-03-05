r"""Contain a cache example generator that caches the last batch and
returns it everytime a batch is generated."""

from __future__ import annotations

__all__ = ["CacheExampleGenerator"]

import copy
from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.example.base import BaseExampleGenerator, setup_example_generator

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch


class CacheExampleGenerator(BaseExampleGenerator):
    r"""Implement an example generator that caches the last batch and
    returns it everytime a batch is generated.

    A new batch is generated only if the batch size changes.

    Args:
        generator: Specifies the example generator or its
            configuration.
        deepcopy: If ``True``, the cached batch is deepcopied before to
            be return.

    Example usage:

    ```pycon
    >>> from startorch.example import Cache, SwissRoll
    >>> generator = Cache(SwissRoll())
    >>> generator
    CacheExampleGenerator(
      (generator): SwissRollExampleGenerator(noise_std=0.0, spin=1.5, hole=False)
      (deepcopy): False
    )
    >>> batch = generator.generate(batch_size=10)
    >>> batch
    {'target': tensor([...]), 'feature': tensor([[...]])}

    ```
    """

    def __init__(self, generator: BaseExampleGenerator | dict, deepcopy: bool = False) -> None:
        self._generator = setup_example_generator(generator)
        self._deepcopy = bool(deepcopy)

        # This variable is used to store the cached value.
        self._cache = None
        self._cache_batch_size = 0

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"generator": self._generator, "deepcopy": self._deepcopy}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        if self._cache is None or self._cache_batch_size != batch_size:
            self._cache = self._generator.generate(batch_size=batch_size, rng=rng)
        tensors = self._cache
        self._cache_batch_size = next(iter(tensors.values())).shape[0]
        if self._deepcopy:
            tensors = copy.deepcopy(tensors)
        return tensors
