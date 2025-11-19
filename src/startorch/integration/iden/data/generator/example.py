r"""Contain example-based data generator implementations."""

from __future__ import annotations

__all__ = ["ExampleDataGenerator"]

from collections.abc import Hashable

import torch
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from startorch.example import BaseExampleGenerator, setup_example_generator
from startorch.utils.imports import is_iden_available
from startorch.utils.seed import get_torch_generator

if is_iden_available():
    from iden.data.generator import BaseDataGenerator
else:
    from startorch.integration.iden.fallback_types import BaseDataGenerator


class ExampleDataGenerator(BaseDataGenerator[dict[Hashable, torch.Tensor]]):
    r"""Implement a data generator to use ``BaseExampleGenerator`` with
    ``iden``.

    Args:
        example: The example generator or its configuration.
        batch_size: The batch size.
        random_seed: The random seed.
    """

    def __init__(
        self, example: BaseExampleGenerator | dict, batch_size: int, random_seed: int
    ) -> None:
        super().__init__()
        self._example = setup_example_generator(example)
        self._batch_size = int(batch_size)
        self._rng = get_torch_generator(random_seed)

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "example": self._example,
                    "batch_size": f"{self._batch_size:,}",
                    "random_seed": self._rng.initial_seed(),
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "example": self._example,
                    "batch_size": f"{self._batch_size:,}",
                    "random_seed": self._rng.initial_seed(),
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self) -> dict[Hashable, torch.Tensor]:
        return self._example.generate(batch_size=self._batch_size, rng=self._rng)
