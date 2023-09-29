from __future__ import annotations

__all__ = [
    "BaseExampleGenerator",
    "Hypercube",
    "HypercubeExampleGenerator",
    "TimeSeries",
    "TimeSeriesExampleGenerator",
    "is_example_generator_config",
    "setup_example_generator",
]

from startorch.example.base import (
    BaseExampleGenerator,
    is_example_generator_config,
    setup_example_generator,
)
from startorch.example.hypercube import HypercubeExampleGenerator
from startorch.example.hypercube import HypercubeExampleGenerator as Hypercube
from startorch.example.timeseries import TimeSeriesExampleGenerator
from startorch.example.timeseries import TimeSeriesExampleGenerator as TimeSeries
