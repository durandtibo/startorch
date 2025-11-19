from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from startorch.plot.matplotlib import hist_feature
from startorch.testing import matplotlib_available
from startorch.utils.imports import is_matplotlib_available

if is_matplotlib_available():
    from matplotlib import pyplot as plt

##################################
#     Tests for hist_feature     #
##################################


@matplotlib_available
@pytest.mark.parametrize(
    "features",
    [
        torch.ones(2, 3),
        np.zeros((2, 3)),
        np.ones((6, 1)),
        np.ones((6, 2)),
        np.ones((6, 6)),
    ],
)
def test_hist_feature(features: torch.Tensor | np.ndarray) -> None:
    assert isinstance(hist_feature(features), plt.Figure)


@matplotlib_available
def test_hist_feature_incorrect_feature_dims_1() -> None:
    with pytest.raises(
        RuntimeError, match=r"Expected a 2D array/tensor but received .* dimensions"
    ):
        hist_feature(np.ones((2,)))


@matplotlib_available
def test_hist_feature_incorrect_feature_dims_3() -> None:
    with pytest.raises(
        RuntimeError, match=r"Expected a 2D array/tensor but received .* dimensions"
    ):
        hist_feature(np.ones((2, 3, 4)))


@matplotlib_available
def test_hist_feature_feature_names() -> None:
    assert isinstance(hist_feature(np.ones((2, 3)), feature_names=["a", "b", "c"]), plt.Figure)


@matplotlib_available
def test_hist_feature_incorrect_not_enough_feature_names() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"The number of features .* does not match with the number of feature names",
    ):
        hist_feature(np.ones((2, 3)), feature_names=["a", "b"])


@matplotlib_available
def test_hist_feature_incorrect_too_many_feature_names() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"The number of features .* does not match with the number of feature names",
    ):
        hist_feature(np.ones((2, 3)), feature_names=["a", "b", "c", "d"])


@patch("startorch.utils.imports.is_matplotlib_available", lambda: False)
def test_hist_feature_no_matplotlib() -> None:
    with pytest.raises(RuntimeError, match=r"'matplotlib' package is required but not installed."):
        hist_feature(np.ones((2, 3)))
