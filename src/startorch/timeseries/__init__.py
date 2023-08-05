from __future__ import annotations

__all__ = [
    "BaseTimeSeriesGenerator",
    "TimeSeries",
    "merge_multiple_timeseries_by_time",
    "setup_timeseries_generator",
    "truncate_timeseries_by_length",
]

from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)
from startorch.timeseries.generic import TimeSeriesGenerator as TimeSeries
from startorch.timeseries.merge import (
    merge_multiple_timeseries_by_time,
    truncate_timeseries_by_length,
)
