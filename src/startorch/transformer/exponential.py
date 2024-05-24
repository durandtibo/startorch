r"""Contain the implementation of tensor transformers that samples
values from an Exponential distribution."""

from __future__ import annotations

__all__ = ["ExponentialTransformer"]


from typing import TYPE_CHECKING

from startorch.random import exponential
from startorch.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:
    import torch


class ExponentialTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that samples values from an
    Exponential distribution.

    The input must be a sequence of tensors with a single item.
    This tensor is interpreted as the rate parameters of the Exponential
    distribution.

    Args:
        rate: The key that contains the rate values of the Exponential
            distribution.
        output: The key that contains the output values sampled from
            the Exponential distribution.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Exponential
    >>> transformer = Exponential(rate="rate", output="output")
    >>> transformer
    ExponentialTransformer(rate=rate, output=output, exist_ok=False)
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
        return exponential(tensor, generator=rng)
