r"""Contain the implementation of tensor generators where the values are
constant."""

from __future__ import annotations

__all__ = ["FullTensorGenerator"]

from typing import TYPE_CHECKING

import torch

from startorch.tensor.base import BaseTensorGenerator

if TYPE_CHECKING:
    from torch import Generator, Tensor


class FullTensorGenerator(BaseTensorGenerator):
    r"""Implement a tensor generator that fills the tensor with a given
    value.

    Args:
        value: Specifies the fill value.
        dtype: Specifies the target dtype. ``None`` means the data type
            is infered from the value type.

    Example usage:

    ```pycon
    >>> from startorch.tensor import Full
    >>> generator = Full(value=42)
    >>> generator
    FullTensorGenerator(value=42, dtype=None)
    >>> generator.generate((2, 6))
    tensor([[42, 42, 42, 42, 42, 42],
            [42, 42, 42, 42, 42, 42]])

    ```
    """

    def __init__(self, value: bool | float, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self._value = value
        self._dtype = dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(value={self._value}, dtype={self._dtype})"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return torch.full(size=size, fill_value=self._value, dtype=self._dtype)
