r"""Contain the implementation of tensor transformers to change the data
type of tensors."""

from __future__ import annotations

__all__ = ["FloatTensorTransformer", "LongTensorTransformer"]

from typing import TYPE_CHECKING

from startorch.tensor.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:
    import torch


class FloatTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transfomer that converts tensor values to
    float.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Float
    >>> transformer = Float()
    >>> transformer
    FloatTensorTransformer()
    >>> transformer.transform(torch.tensor([[1, -2, 3], [-4, 5, -6]]))
    tensor([[ 1., -2.,  3.], [-4.,  5., -6.]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.float()


class LongTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transfomer that converts a tensor values to
    long.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Long
    >>> transformer = Long()
    >>> transformer
    LongTensorTransformer()
    >>> transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    tensor([[ 1, -2,  3], [-4,  5, -6]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.long()
