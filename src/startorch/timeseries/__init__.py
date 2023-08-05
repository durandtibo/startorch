from __future__ import annotations

__all__ = ["BaseTimeSeriesGenerator", "TimeSeries", "setup_timeseries_generator"]

from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)
from startorch.timeseries.generic import TimeSeriesGenerator as TimeSeries
