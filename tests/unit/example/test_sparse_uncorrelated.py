from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example import make_sparse_uncorrelated_regression
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


#########################################################
#     Tests for make_sparse_uncorrelated_regression     #
#########################################################


@pytest.mark.parametrize("num_examples", [0, -1])
def test_make_sparse_uncorrelated_regression_incorrect_num_examples(num_examples: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for num_examples. Expected a value greater or equal to 1",
    ):
        make_sparse_uncorrelated_regression(num_examples=num_examples)


@pytest.mark.parametrize("feature_size", [3, 1, 0, -1])
def test_make_sparse_uncorrelated_regression_incorrect_feature_size(feature_size: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for feature_size. Expected a value greater or equal to 4",
    ):
        make_sparse_uncorrelated_regression(feature_size=feature_size)


@pytest.mark.parametrize("noise_std", [-1, -4.2])
def test_make_sparse_uncorrelated_regression_incorrect_noise_std(noise_std: float) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        make_sparse_uncorrelated_regression(noise_std=noise_std)


def test_make_sparse_uncorrelated_regression() -> None:
    data = make_sparse_uncorrelated_regression(num_examples=10)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    features = data[ct.FEATURE]
    assert isinstance(features, BatchedTensor)
    assert features.shape == (10, 4)
    assert features.dtype == torch.float


def test_make_sparse_uncorrelated_regression_feature_size_8() -> None:
    data = make_sparse_uncorrelated_regression(num_examples=10, feature_size=8)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    features = data[ct.FEATURE]
    assert isinstance(features, BatchedTensor)
    assert features.shape == (10, 8)
    assert features.dtype == torch.float


@pytest.mark.parametrize("num_examples", SIZES)
def test_make_sparse_uncorrelated_regression_num_examples(num_examples: int) -> None:
    data = make_sparse_uncorrelated_regression(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


@pytest.mark.parametrize("feature_size", [5, 8, 10])
def test_make_sparse_uncorrelated_regression_feature_size(feature_size: int) -> None:
    data = make_sparse_uncorrelated_regression(num_examples=10, feature_size=feature_size)
    assert data[ct.FEATURE].shape[1] == feature_size


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_make_sparse_uncorrelated_regression_same_random_seed(noise_std: float) -> None:
    assert objects_are_equal(
        make_sparse_uncorrelated_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_sparse_uncorrelated_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
    )


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_make_sparse_uncorrelated_regression_different_random_seeds(noise_std: float) -> None:
    assert not objects_are_equal(
        make_sparse_uncorrelated_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_sparse_uncorrelated_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(2),
        ),
    )
