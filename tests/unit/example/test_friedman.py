from __future__ import annotations

import torch
from coola import objects_are_equal
from pytest import mark, raises
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example import make_friedman1
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


####################################
#     Tests for make_friedman1     #
####################################


@mark.parametrize("num_examples", (0, -1))
def test_make_friedman1_incorrect_num_examples(num_examples: int) -> None:
    with raises(RuntimeError, match="The number of examples .* has to be greater than 0"):
        make_friedman1(num_examples=num_examples)


@mark.parametrize("feature_size", (4, 1, 0, -1))
def test_make_friedman1_incorrect_feature_size(feature_size: int) -> None:
    with raises(RuntimeError, match="feature_size (.*) has to be greater or equal to 5"):
        make_friedman1(feature_size=feature_size)


def test_make_friedman1_incorrect_noise_std() -> None:
    with raises(
        RuntimeError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        make_friedman1(noise_std=-1)


def test_make_friedman1() -> None:
    data = make_friedman1(num_examples=10, feature_size=8)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].shape == (10, 8)
    assert data[ct.FEATURE].dtype == torch.float


@mark.parametrize("num_examples", SIZES)
def test_make_friedman1_num_examples(num_examples: int) -> None:
    data = make_friedman1(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


@mark.parametrize("feature_size", (5, 8, 10))
def test_make_friedman1_feature_size(feature_size: int) -> None:
    data = make_friedman1(num_examples=10, feature_size=feature_size)
    assert data[ct.FEATURE].shape[1] == feature_size


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_friedman1_same_random_seed(noise_std: float) -> None:
    assert objects_are_equal(
        make_friedman1(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_friedman1(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
    )


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_friedman1_different_random_seeds(noise_std: float) -> None:
    assert not objects_are_equal(
        make_friedman1(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_friedman1(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(2),
        ),
    )
