from __future__ import annotations

__all__ = [
    "BaseTimeSeriesGenerator",
    "Merge",
    "MergeTimeSeriesGenerator",
    "MixedTimeSeries",
    "MixedTimeSeriesGenerator",
    "MultinomialChoice",
    "MultinomialChoiceTimeSeriesGenerator",
    "Periodic",
    "PeriodicTimeSeriesGenerator",
    "TimeSeries",
    "TimeSeriesGenerator",
    "is_timeseries_generator_config",
    "setup_timeseries_generator",
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
from startorch.timeseries.merge import MergeTimeSeriesGenerator
from startorch.timeseries.merge import MergeTimeSeriesGenerator as Merge
from startorch.timeseries.mixed import MixedTimeSeriesGenerator
from startorch.timeseries.mixed import MixedTimeSeriesGenerator as MixedTimeSeries
from startorch.timeseries.periodic import PeriodicTimeSeriesGenerator
from startorch.timeseries.periodic import PeriodicTimeSeriesGenerator as Periodic
