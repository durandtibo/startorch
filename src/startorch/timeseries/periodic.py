r"""Implement a time series generator to generate periodic time series
from a regular time series generator."""

from __future__ import annotations

__all__ = ["PeriodicTimeSeriesGenerator"]

import math
from typing import TYPE_CHECKING

from batchtensor.nested import repeat_along_seq, slice_along_seq
from coola.utils import str_indent, str_mapping

from startorch.periodic.timeseries.base import BasePeriodicTimeSeriesGenerator
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator
from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch


class PeriodicTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implement a time series generator to generate periodic time
    series from a regular time series generator.

    Args:
        timeseries: Specifies a time series generator or its
            configuration that is used to generate the periodic
            pattern.
        period: Specifies the period length sampler or its
            configuration. This sampler is used to sample the period
            length at each batch.

    Example usage:

    ```pycon

    >>> from startorch.timeseries import Periodic, TimeSeries
    >>> from startorch.sequence import RandUniform
    >>> from startorch.tensor import RandInt
    >>> generator = Periodic(
    ...     TimeSeries({"value": RandUniform(), "time": RandUniform()}), period=RandInt(2, 5)
    ... )
    >>> generator
    PeriodicTimeSeriesGenerator(
      (sequence): TimeSeriesGenerator(
          (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
          (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
      (period): RandIntTensorGenerator(low=2, high=5)
    )
    >>> generator.generate(seq_len=10, batch_size=2)
    {'value': tensor([[...]]), 'time': tensor([[...]])}

    ```
    """

    def __init__(
        self,
        timeseries: BaseTimeSeriesGenerator | BasePeriodicTimeSeriesGenerator | dict,
        period: BaseTensorGenerator | dict,
    ) -> None:
        super().__init__()
        self._timeseries = setup_timeseries_generator(timeseries)
        self._period = setup_tensor_generator(period)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"sequence": self._timeseries, "period": self._period}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        period = int(self._period.generate((1,), rng=rng).item())
        if isinstance(self._timeseries, BasePeriodicTimeSeriesGenerator):
            return self._timeseries.generate(
                seq_len=seq_len, period=period, batch_size=batch_size, rng=rng
            )
        data = self._timeseries.generate(seq_len=period, batch_size=batch_size, rng=rng)
        data = repeat_along_seq(data, math.ceil(seq_len / period))
        return slice_along_seq(data, stop=seq_len)
