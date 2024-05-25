r"""Contain the implementation of a time series generator that generates
time series from tensor generators."""

from __future__ import annotations

__all__ = ["TensorTimeSeriesGenerator"]


from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator
from startorch.timeseries.base import BaseTimeSeriesGenerator

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    import torch


class TensorTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implement a time series generator that generates time series from
    tensor generators.

    Args:
        generators: The tensor generators or their configurations.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor import RandUniform
    >>> from startorch.timeseries import TensorTimeSeriesGenerator
    >>> generator = TensorTimeSeriesGenerator({"value": RandUniform(), "time": RandUniform()})
    >>> generator
    TensorTimeSeriesGenerator(
      (value): RandUniformTensorGenerator(low=0.0, high=1.0)
      (time): RandUniformTensorGenerator(low=0.0, high=1.0)
      (size): ()
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    {'value': tensor([[...]]), 'time': tensor([[...]])}

    ```
    """

    def __init__(
        self,
        generators: Mapping[str, BaseTensorGenerator | dict],
        size: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self._generators = {
            key: setup_tensor_generator(generator) for key, generator in generators.items()
        }
        self._size = tuple(size)

    def __repr__(self) -> str:
        args = str_indent(str_mapping(self._generators | {"size": self._size}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        size = (batch_size, seq_len, *self._size)
        return {
            key: generator.generate(size=size, rng=rng)
            for key, generator in self._generators.items()
        }
