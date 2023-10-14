from __future__ import annotations

__all__ = [
    "BaseExampleGenerator",
    "Hypercube",
    "HypercubeExampleGenerator",
    "LinearRegression",
    "LinearRegressionExampleGenerator",
    "SwissRoll",
    "SwissRollExampleGenerator",
    "TimeSeries",
    "TimeSeriesExampleGenerator",
    "is_example_generator_config",
    "make_linear_regression",
    "make_swiss_roll",
    "setup_example_generator",
]

from startorch.example.base import (
    BaseExampleGenerator,
    is_example_generator_config,
    setup_example_generator,
)
from startorch.example.hypercube import HypercubeExampleGenerator
from startorch.example.hypercube import HypercubeExampleGenerator as Hypercube
from startorch.example.regression import LinearRegressionExampleGenerator
from startorch.example.regression import (
    LinearRegressionExampleGenerator as LinearRegression,
)
from startorch.example.regression import make_linear_regression
from startorch.example.swissroll import SwissRollExampleGenerator
from startorch.example.swissroll import SwissRollExampleGenerator as SwissRoll
from startorch.example.swissroll import make_swiss_roll
from startorch.example.timeseries import TimeSeriesExampleGenerator
from startorch.example.timeseries import TimeSeriesExampleGenerator as TimeSeries
