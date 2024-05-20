r"""Contain the implementation of tensor transformers that computes
trigonometric functions on tensors."""

from __future__ import annotations

__all__ = ["PoissonTransformer"]


import torch

from startorch.transformer.base import BaseTensorTransformer


class PoissonTransformer(BaseTensorTransformer):
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
    >>> from startorch.transformer import Poisson
    >>> transformer = Poisson(rate="rate", output="output")
    >>> transformer
    PoissonTransformer(rate=rate, output=output, exist_ok=False)
    >>> data = {"rate": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'rate': tensor([[1., 2., 3.],
                     [4., 5., 6.]]),
     'output': tensor([[...]])}


    ```
    """

    def __init__(self, rate: str, output: str, exist_ok: bool = False) -> None:
        super().__init__(input=rate, output=output, exist_ok=exist_ok)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(rate={self._input}, "
            f"output={self._output}, exist_ok={self._exist_ok})"
        )

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,
    ) -> torch.Tensor:
        return torch.poisson(tensor, generator=rng)
