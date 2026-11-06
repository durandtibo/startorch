r"""Contain some periodic time series generators."""

__all__ = [
    "BasePeriodicTimeSeriesGenerator",
    "Repeat",
    "RepeatPeriodicTimeSeriesGenerator",
    "is_periodic_timeseries_generator_config",
    "setup_periodic_timeseries_generator",
]

from startorch.periodic.timeseries.base import (
    BasePeriodicTimeSeriesGenerator,
    is_periodic_timeseries_generator_config,
    setup_periodic_timeseries_generator,
)
from startorch.periodic.timeseries.repeat import (
    RepeatPeriodicTimeSeriesGenerator,
    RepeatPeriodicTimeSeriesGenerator as Repeat,
)
