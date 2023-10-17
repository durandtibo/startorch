from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch
from pytest import mark, raises

from startorch.plot.matplotlib import plot_features
from startorch.testing import matplotlib_available
from startorch.utils.imports import is_matplotlib_available

if is_matplotlib_available():
    from matplotlib import pyplot as plt

###################################
#     Tests for plot_features     #
###################################


@matplotlib_available
@mark.parametrize(
    "features",
    (torch.ones(2, 3), np.zeros((2, 3)), np.ones((6, 1)), np.ones((6, 2)), np.ones((6, 6))),
)
def test_plot_features(features: torch.Tensor | np.ndarray) -> None:
    assert isinstance(plot_features(features), plt.Figure)


@matplotlib_available
def test_plot_features_incorrect_feature_dims_1() -> None:
    with raises(RuntimeError, match="Expected a 2D array/tensor but received .* dimensions"):
        plot_features(np.ones((2,)))


@matplotlib_available
def test_plot_features_incorrect_feature_dims_3() -> None:
    with raises(RuntimeError, match="Expected a 2D array/tensor but received .* dimensions"):
        plot_features(np.ones((2, 3, 4)))


@matplotlib_available
def test_plot_features_feature_names() -> None:
    assert isinstance(plot_features(np.ones((2, 3)), feature_names=["a", "b", "c"]), plt.Figure)


@matplotlib_available
def test_plot_features_incorrect_not_enough_feature_names() -> None:
    with raises(
        RuntimeError,
        match="The number of features .* does not match with the number of feature names",
    ):
        plot_features(np.ones((2, 3)), feature_names=["a", "b"])


@matplotlib_available
def test_plot_features_incorrect_too_many_feature_names() -> None:
    with raises(
        RuntimeError,
        match="The number of features .* does not match with the number of feature names",
    ):
        plot_features(np.ones((2, 3)), feature_names=["a", "b", "c", "d"])


@patch("startorch.utils.imports.is_matplotlib_available", lambda *args, **kwargs: False)
def test_plot_features_no_matplotlib() -> None:
    with raises(RuntimeError, match="`matplotlib` package is required but not installed."):
        plot_features(np.ones((2, 3)))
