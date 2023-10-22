from __future__ import annotations

__all__ = ["CacheExampleGenerator"]


import torch
from coola.utils import str_indent, str_mapping
from redcat import BatchDict

from startorch.example.base import BaseExampleGenerator, setup_example_generator


class CacheExampleGenerator(BaseExampleGenerator):
    r"""Implements an example generator that caches the last batch and
    returns it everytime a batch is generated.

    A new batch is generated only if the batch size changes.

    Args:
    ----
        generator (``BaseExampleGenerator`` or dict): Specifies the
            example generator or its configuration.
        deepcopy (bool, optional): If ``True``, the cached batch is
            deepcopied before to be return. Default: ``True``

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import Cache, SwissRoll
        >>> generator = Cache(SwissRoll())
        >>> generator
        CacheExampleGenerator(
          (generator): SwissRollExampleGenerator(noise_std=0.0, spin=1.5, hole=False)
          (deepcopy): False
        )
        >>> batch = generator.generate(batch_size=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """

    def __init__(self, generator: BaseExampleGenerator | dict, deepcopy: bool = False) -> None:
        self._generator = setup_example_generator(generator)
        self._deepcopy = bool(deepcopy)

        # This variable is used to store the cached value.
        self._cache = None

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"generator": self._generator, "deepcopy": self._deepcopy}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, batch_size: int = 1, rng: torch.Generator | None = None) -> BatchDict:
        if self._cache is None or self._cache.batch_size != batch_size:
            self._cache = self._generator.generate(batch_size=batch_size, rng=rng)
        batch = self._cache
        if self._deepcopy:
            batch = batch.clone()
        return batch
