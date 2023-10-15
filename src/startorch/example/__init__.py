from __future__ import annotations

__all__ = [
    "BaseExampleGenerator",
    "HypercubeClassification",
    "HypercubeClassificationExampleGenerator",
    "LinearRegression",
    "LinearRegressionExampleGenerator",
    "SwissRoll",
    "SwissRollExampleGenerator",
    "TimeSeries",
    "TimeSeriesExampleGenerator",
    "is_example_generator_config",
    "make_friedman1",
    "make_friedman2",
    "make_friedman3",
    "make_hypercube_classification",
    "make_linear_regression",
    "make_swiss_roll",
    "setup_example_generator",
    "Friedman1Regression",
    "Friedman1RegressionExampleGenerator",
]

from startorch.example.base import (
    BaseExampleGenerator,
    is_example_generator_config,
    setup_example_generator,
)
from startorch.example.friedman import Friedman1RegressionExampleGenerator
from startorch.example.friedman import (
    Friedman1RegressionExampleGenerator as Friedman1Regression,
)
from startorch.example.friedman import make_friedman1, make_friedman2, make_friedman3
from startorch.example.hypercube import HypercubeClassificationExampleGenerator
from startorch.example.hypercube import (
    HypercubeClassificationExampleGenerator as HypercubeClassification,
)
from startorch.example.hypercube import make_hypercube_classification
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
