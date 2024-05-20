r"""Contain the implementation of tensor transformers that computes
trigonometric functions on tensors."""

from __future__ import annotations

__all__ = ["PoissonTensorTransformer"]


import torch

from startorch.transformer.tensor.base import BaseTensorTransformer


class PoissonTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that samples values from a Poisson
    distribution.

    The input must be a sequence of tensors with a single item.
    This tensor is interpreted as the rate parameters of the Poisson
    distribution.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer.tensor import Poisson
    >>> transformer = Poisson(rate="rate", output="output")
    >>> transformer
    PoissonTensorTransformer(rate=rate, output=output)
    >>> data = {"rate": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'rate': tensor([[1., 2., 3.],
                     [4., 5., 6.]]),
     'output': tensor([[...]])}


    ```
    """

    def __init__(self, rate: str, output: str) -> None:
        self._rate = rate
        self._output = output

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(rate={self._rate}, output={self._output})"

    def transform(
        self,
        data: dict[str, torch.Tensor],
        *,
        rng: torch.Transformer | None = None,
    ) -> dict[str, torch.Tensor]:
        data = data.copy()
        data[self._output] = torch.poisson(data[self._rate], generator=rng)
        return data
