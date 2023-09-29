from __future__ import annotations

__all__ = [
    "BaseTimeSeriesGenerator",
    "MultinomialChoice",
    "MultinomialChoiceTimeSeriesGenerator",
    "Periodic",
    "PeriodicTimeSeriesGenerator",
    "TimeSeries",
    "TimeSeriesGenerator",
    "is_timeseries_generator_config",
    "merge_multiple_timeseries_by_time",
    "setup_timeseries_generator",
    "truncate_timeseries_by_length",
]

from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    is_timeseries_generator_config,
    setup_timeseries_generator,
)
from startorch.timeseries.choice import MultinomialChoiceTimeSeriesGenerator
from startorch.timeseries.choice import (
    MultinomialChoiceTimeSeriesGenerator as MultinomialChoice,
)
from startorch.timeseries.generic import TimeSeriesGenerator
from startorch.timeseries.generic import TimeSeriesGenerator as TimeSeries
from startorch.timeseries.merge import (
    merge_multiple_timeseries_by_time,
    truncate_timeseries_by_length,
)
from startorch.timeseries.periodic import PeriodicTimeSeriesGenerator
from startorch.timeseries.periodic import PeriodicTimeSeriesGenerator as Periodic
