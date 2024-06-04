r"""Contain the implementation of tensor transformers that samples
values from an Poisson distribution."""

from __future__ import annotations

__all__ = ["PoissonTensorTransformer"]


from typing import TYPE_CHECKING

from torch import poisson

from startorch.tensor.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:
    import torch


class PoissonTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that samples values from an
    Poisson distribution.

    The input tensor is interpreted as the rate parameters of the Poisson
    distribution.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Poisson
    >>> transformer = Poisson()
    >>> transformer
    PoissonTensorTransformer()
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
        return poisson(tensor, generator=rng)
