r"""Contain arithmetic tensor transformers."""

from __future__ import annotations

__all__ = ["AddTensorTransformer", "MulTensorTransformer"]

from typing import TYPE_CHECKING

from startorch.tensor.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:
    import torch


class AddTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that adds a scalar value to the
    input tensor.

    This tensor transformer is equivalent to: ``output = input + value``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Add
    >>> transformer = Add(1)
    >>> transformer
    AddTensorTransformer(value=1)
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[ 2., -1.,  4.], [-3.,  6., -5.]])

    ```
    """

    def __init__(self, value: float) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(value={self._value})"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.add(self._value)


class MulTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that multiplies a scalar value to
    the input tensor.

    This tensor transformer is equivalent to: ``output = input * value``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Mul
    >>> transformer = Mul(2)
    >>> transformer
    MulTensorTransformer(value=2)
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[  2.,  -4.,   6.], [ -8.,  10., -12.]])

    ```
    """

    def __init__(self, value: float) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(value={self._value})"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.mul(self._value)
