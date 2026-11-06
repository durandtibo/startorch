r"""Contain example generators."""

from __future__ import annotations

__all__ = [
    "BaseExampleGenerator",
    "BlobsClassification",
    "BlobsClassificationExampleGenerator",
    "Cache",
    "CacheExampleGenerator",
    "CirclesClassification",
    "CirclesClassificationExampleGenerator",
    "Concatenate",
    "ConcatenateExampleGenerator",
    "Friedman1Regression",
    "Friedman1RegressionExampleGenerator",
    "Friedman2Regression",
    "Friedman2RegressionExampleGenerator",
    "Friedman3Regression",
    "Friedman3RegressionExampleGenerator",
    "HypercubeClassification",
    "HypercubeClassificationExampleGenerator",
    "LinearRegression",
    "LinearRegressionExampleGenerator",
    "MoonsClassification",
    "MoonsClassificationExampleGenerator",
    "SwissRoll",
    "SwissRollExampleGenerator",
    "TensorExampleGenerator",
    "TimeSeriesExampleGenerator",
    "TransformExampleGenerator",
    "VanillaExampleGenerator",
    "is_example_generator_config",
    "make_blobs_classification",
    "make_circles_classification",
    "make_friedman1_regression",
    "make_friedman2_regression",
    "make_friedman3_regression",
    "make_hypercube_classification",
    "make_linear_regression",
    "make_moons_classification",
    "make_sparse_uncorrelated_regression",
    "make_swiss_roll",
    "setup_example_generator",
]

from startorch.example.base import (
    BaseExampleGenerator,
    is_example_generator_config,
    setup_example_generator,
)
from startorch.example.blobs import (
    BlobsClassificationExampleGenerator,
    BlobsClassificationExampleGenerator as BlobsClassification,
    make_blobs_classification,
)
from startorch.example.cache import CacheExampleGenerator, CacheExampleGenerator as Cache
from startorch.example.circles import (
    CirclesClassificationExampleGenerator,
    CirclesClassificationExampleGenerator as CirclesClassification,
    make_circles_classification,
)
from startorch.example.concatenate import (
    ConcatenateExampleGenerator,
    ConcatenateExampleGenerator as Concatenate,
)
from startorch.example.friedman import (
    Friedman1RegressionExampleGenerator,
    Friedman1RegressionExampleGenerator as Friedman1Regression,
    Friedman2RegressionExampleGenerator,
    Friedman2RegressionExampleGenerator as Friedman2Regression,
    Friedman3RegressionExampleGenerator,
    Friedman3RegressionExampleGenerator as Friedman3Regression,
    make_friedman1_regression,
    make_friedman2_regression,
    make_friedman3_regression,
)
from startorch.example.hypercube import (
    HypercubeClassificationExampleGenerator,
    HypercubeClassificationExampleGenerator as HypercubeClassification,
    make_hypercube_classification,
)
from startorch.example.moons import (
    MoonsClassificationExampleGenerator,
    MoonsClassificationExampleGenerator as MoonsClassification,
    make_moons_classification,
)
from startorch.example.regression import (
    LinearRegressionExampleGenerator,
    LinearRegressionExampleGenerator as LinearRegression,
    make_linear_regression,
)
from startorch.example.sparse_uncorrelated import make_sparse_uncorrelated_regression
from startorch.example.swissroll import (
    SwissRollExampleGenerator,
    SwissRollExampleGenerator as SwissRoll,
    make_swiss_roll,
)
from startorch.example.tensor import TensorExampleGenerator
from startorch.example.timeseries import TimeSeriesExampleGenerator
from startorch.example.transform import TransformExampleGenerator
from startorch.example.vanilla import VanillaExampleGenerator
