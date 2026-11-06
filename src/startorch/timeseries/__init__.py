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
    "TransformTimeSeriesGenerator",
    "VanillaTimeSeriesGenerator",
    "is_timeseries_generator_config",
    "setup_timeseries_generator",
]

from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    is_timeseries_generator_config,
    setup_timeseries_generator,
)
from startorch.timeseries.choice import (
    MultinomialChoiceTimeSeriesGenerator,
    MultinomialChoiceTimeSeriesGenerator as MultinomialChoice,
)
from startorch.timeseries.concatenate import (
    ConcatenateTimeSeriesGenerator,
    ConcatenateTimeSeriesGenerator as Concatenate,
)
from startorch.timeseries.merge import MergeTimeSeriesGenerator, MergeTimeSeriesGenerator as Merge
from startorch.timeseries.mixed import (
    MixedTimeSeriesGenerator,
    MixedTimeSeriesGenerator as MixedTimeSeries,
)
from startorch.timeseries.periodic import (
    PeriodicTimeSeriesGenerator,
    PeriodicTimeSeriesGenerator as Periodic,
)
from startorch.timeseries.sequence import (
    SequenceTimeSeriesGenerator,
    SequenceTimeSeriesGenerator as SequenceTimeSeries,
)
from startorch.timeseries.tensor import (
    TensorTimeSeriesGenerator,
    TensorTimeSeriesGenerator as TensorTimeSeries,
)
from startorch.timeseries.transform import TransformTimeSeriesGenerator
from startorch.timeseries.vanilla import VanillaTimeSeriesGenerator
