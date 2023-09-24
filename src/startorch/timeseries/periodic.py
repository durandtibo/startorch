from __future__ import annotations

__all__ = ["PeriodicTimeSeriesGenerator"]

import math

from coola.utils import str_indent, str_mapping
from redcat import BatchDict
from torch import Generator

from startorch.periodic.timeseries.base import BasePeriodicTimeSeriesGenerator
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator
from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)


class PeriodicTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implements a time series generator to generate periodic time
    series from a regular time series generator.

    Args:
    ----
        timeseries (``BaseTimeSeriesGenerator`` or
            ``BasePeriodicTimeSeriesGenerator`` or dict): Specifies a
            time series generator or its configuration that is used to
            generate the periodic pattern.
        period (``BaseTensorGenerator`` or dict): Specifies
            the period length sampler or its configuration. This
            sampler is used to sample the period length at each
            batch.

    Example usage:

    .. code-block:: pycon

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
        BatchDict(
          (value): tensor([[...]], batch_dim=0, seq_dim=1)
          (time): tensor([[...]], batch_dim=0, seq_dim=1)
        )
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
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchDict:
        period = int(self._period.generate((1,), rng=rng).item())
        if isinstance(self._timeseries, BasePeriodicTimeSeriesGenerator):
            return self._timeseries.generate(
                seq_len=seq_len, period=period, batch_size=batch_size, rng=rng
            )
        return (
            self._timeseries.generate(seq_len=period, batch_size=batch_size, rng=rng)
            .repeat_along_seq(math.ceil(seq_len / period))
            .slice_along_seq(stop=seq_len)
        )
