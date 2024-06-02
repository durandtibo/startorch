r"""Contain an example generator to "generate" the input data."""

from __future__ import annotations

__all__ = ["VanillaTimeSeriesGenerator"]

from typing import TYPE_CHECKING

from batchtensor.nested import slice_along_batch, slice_along_seq

from startorch.timeseries.base import BaseTimeSeriesGenerator

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch


class VanillaTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implement a time series generator to "generate" the input data.

    Args:
        data: The time series data to generate. The dictionary cannot be empty.

    Raises:
        ValueError: if ``data`` is an empty dictionary.

    TimeSeries usage:

    ```pycon

    >>> import torch
    >>> from startorch.timeseries import VanillaTimeSeriesGenerator
    >>> generator = VanillaTimeSeriesGenerator(
    ...     data={"value": torch.ones(4, 10), "time": torch.arange(40).view(4, 10)}
    ... )
    >>> generator
    VanillaTimeSeriesGenerator(batch_size=4, seq_len=10)
    >>> generator.generate(batch_size=3, seq_len=6)
    {'value': tensor([[1., 1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1.]]),
     'time': tensor([[ 0,  1,  2,  3,  4,  5],
                     [10, 11, 12, 13, 14, 15],
                     [20, 21, 22, 23, 24, 25]])}

    ```
    """

    def __init__(self, data: dict[Hashable, torch.Tensor]) -> None:
        super().__init__()
        if not data:
            msg = "data cannot be empty"
            raise ValueError(msg)
        self._data = data
        tensor = next(iter(data.values()))
        self._batch_size = tensor.shape[0]
        self._seq_len = tensor.shape[1]

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(batch_size={self._batch_size:,}, seq_len={self._seq_len:,})"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None  # noqa: ARG002
    ) -> dict[Hashable, torch.Tensor]:
        if batch_size > self._batch_size:
            msg = (
                f"Incorrect batch_size: {batch_size:,}. "
                f"batch_size cannot be greater than {self._batch_size:,}"
            )
            raise RuntimeError(msg)
        if seq_len > self._seq_len:
            msg = (
                f"Incorrect seq_len: {seq_len:,}. "
                f"seq_len cannot be greater than {self._seq_len:,}"
            )
            raise RuntimeError(msg)
        data = slice_along_batch(self._data, stop=batch_size)
        return slice_along_seq(data, stop=seq_len)
