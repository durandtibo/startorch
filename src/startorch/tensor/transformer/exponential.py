r"""Contain the implementation of tensor transformers that samples
values from an Exponential distribution."""

from __future__ import annotations

__all__ = ["ExponentialTensorTransformer"]


from typing import TYPE_CHECKING

from startorch.random import exponential
from startorch.tensor.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:
    import torch


class ExponentialTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that samples values from an
    Exponential distribution.

    The input tensor is interpreted as the rate parameters of the Exponential
    distribution.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Exponential
    >>> transformer = Exponential()
    >>> transformer
    ExponentialTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    >>> out
    tensor([[...]])


    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,
    ) -> torch.Tensor:
        return exponential(tensor, generator=rng)
