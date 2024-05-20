r"""Contain the implementation of tensor transformers that computes
trigonometric functions on tensors."""

from __future__ import annotations

__all__ = ["PoissonTensorTransformer"]


from typing import TYPE_CHECKING

import torch

from startorch.transformer.tensor.base import BaseTensorTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence


class PoissonTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that samples values from a Poisson
    distribution.

    The input values are used as the rate parameters of the Poisson distribution.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer.tensor import Poisson
    >>> transformer = Poisson()
    >>> transformer
    PoissonTensorTransformer()
    >>> rate = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> out = transformer.transform([rate])
    >>> out
    tensor([[...]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self, tensors: Sequence[torch.Tensor], *, rng: torch.Transformer | None = None
    ) -> torch.Tensor:
        (rate,) = tensors
        return torch.poisson(rate, generator=rng)
