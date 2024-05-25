r"""Contain time series generators."""

from __future__ import annotations

__all__ = [
    "BaseTimeSeriesGenerator",
    "Concatenate",
    "ConcatenateTimeSeriesGenerator",
    "Merge",
    "MergeTimeSeriesGenerator",
    "MixedTimeSeries",
    "MixedTimeSeriesGenerator",
    "MultinomialChoice",
    "MultinomialChoiceTimeSeriesGenerator",
    "Periodic",
    "PeriodicTimeSeriesGenerator",
    "SequenceTimeSeries",
    "SequenceTimeSeriesGenerator",
    "TensorTimeSeries",
    "TensorTimeSeriesGenerator",
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
from startorch.timeseries.concatenate import ConcatenateTimeSeriesGenerator
from startorch.timeseries.concatenate import (
    ConcatenateTimeSeriesGenerator as Concatenate,
)
from startorch.timeseries.merge import MergeTimeSeriesGenerator
from startorch.timeseries.merge import MergeTimeSeriesGenerator as Merge
from startorch.timeseries.mixed import MixedTimeSeriesGenerator
from startorch.timeseries.mixed import MixedTimeSeriesGenerator as MixedTimeSeries
from startorch.timeseries.periodic import PeriodicTimeSeriesGenerator
from startorch.timeseries.periodic import PeriodicTimeSeriesGenerator as Periodic
from startorch.timeseries.sequence import SequenceTimeSeriesGenerator
from startorch.timeseries.sequence import (
    SequenceTimeSeriesGenerator as SequenceTimeSeries,
)
from startorch.timeseries.tensor import TensorTimeSeriesGenerator
from startorch.timeseries.tensor import TensorTimeSeriesGenerator as TensorTimeSeries
