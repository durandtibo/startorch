from __future__ import annotations

__all__ = ["CacheExampleGenerator"]


import torch
from coola.utils import str_indent, str_mapping
from redcat import BatchDict

from startorch.example.base import BaseExampleGenerator, setup_example_generator


class CacheExampleGenerator(BaseExampleGenerator):
    r"""Implements.

    Args:
    ----
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise. Default: ``0.0``
        spin (float or int, optional): Specifies the number of spins
            of the Swiss roll. Default: ``1.5``
        hole (bool, optional): If ``True`` generates the Swiss roll
            with hole dataset. Default: ``False``

    Raises:
    ------
        ValueError if one of the parameters is not valid.

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
