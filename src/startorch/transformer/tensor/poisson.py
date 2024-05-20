r"""Contain the implementation of tensor transformers that computes
trigonometric functions on tensors."""

from __future__ import annotations

__all__ = ["PoissonTensorTransformer"]


import torch

from startorch.transformer.tensor.base import BaseTensorTransformer
from startorch.transformer.tensor.utils import add_item, check_input_keys


class PoissonTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that samples values from a Poisson
    distribution.

    The input must be a sequence of tensors with a single item.
    This tensor is interpreted as the rate parameters of the Poisson
    distribution.

    Args:
        rate: The key that contains the rate values of the Poisson
            distribution.
        output: The key that contains the output values sampled from
            the Poisson distribution.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer.tensor import Poisson
    >>> transformer = Poisson(rate="rate", output="output")
    >>> transformer
    PoissonTensorTransformer(rate=rate, output=output, exist_ok=False)
    >>> data = {"rate": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'rate': tensor([[1., 2., 3.],
                     [4., 5., 6.]]),
     'output': tensor([[...]])}


    ```
    """

    def __init__(self, rate: str, output: str, exist_ok: bool = False) -> None:
        self._rate = rate
        self._output = output
        self._exist_ok = exist_ok

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(rate={self._rate}, "
            f"output={self._output}, exist_ok={self._exist_ok})"
        )

    def transform(
        self,
        data: dict[str, torch.Tensor],
        *,
        rng: torch.Transformer | None = None,
    ) -> dict[str, torch.Tensor]:
        check_input_keys(data, keys=[self._rate])
        data = data.copy()
        add_item(
            data,
            key=self._output,
            value=torch.poisson(data[self._rate], generator=rng),
            exist_ok=self._exist_ok,
        )
        return data
