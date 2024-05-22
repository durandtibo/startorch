r"""Contain an example generator to "generate" the input data."""

from __future__ import annotations

__all__ = ["VanillaExampleGenerator"]

from typing import TYPE_CHECKING

from batchtensor.nested import slice_along_batch

from startorch.example.base import BaseExampleGenerator

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch


class VanillaExampleGenerator(BaseExampleGenerator):
    r"""Implement an example generator to "generate" the input data.

    Args:
        data: The data to generate. The dictionary cannot be empty.

    Raises:
        ValueError: if ``data`` is an empty dictionary.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.example import VanillaExampleGenerator
    >>> generator = VanillaExampleGenerator(
    ...     data={"value": torch.ones(10, 3), "time": torch.arange(10)}
    ... )
    >>> generator
    VanillaExampleGenerator(batch_size=10)
    >>> generator.generate(batch_size=5)
    {'value': tensor([[1., 1., 1.],
                      [1., 1., 1.],
                      [1., 1., 1.],
                      [1., 1., 1.],
                      [1., 1., 1.]]),
     'time': tensor([0, 1, 2, 3, 4])}

    ```
    """

    def __init__(self, data: dict[Hashable, torch.Tensor]) -> None:
        super().__init__()
        if not data:
            msg = "data cannot be empty"
            raise ValueError(msg)
        self._data = data
        self._batch_size = next(iter(data.values())).shape[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(batch_size={self._batch_size})"

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None  # noqa: ARG002
    ) -> dict[Hashable, torch.Tensor]:
        if batch_size > self._batch_size:
            msg = (
                f"Incorrect batch_size: {batch_size:,}. "
                f"batch_size cannot be greater than {self._batch_size:,}"
            )
            raise RuntimeError(msg)
        return slice_along_batch(self._data, stop=batch_size)
