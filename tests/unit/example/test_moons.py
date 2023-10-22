from __future__ import annotations

import torch
from coola import objects_are_equal
from pytest import mark, raises
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example import make_moons_classification
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


###############################################
#     Tests for make_moons_classification     #
###############################################


@mark.parametrize("num_examples", (0, -1))
def test_make_moons_classification_incorrect_num_examples(num_examples: int) -> None:
    with raises(
        RuntimeError,
        match="Incorrect value for num_examples. Expected a value greater or equal to 1",
    ):
        make_moons_classification(num_examples=num_examples)


@mark.parametrize("noise_std", (-1, -4.2))
def test_make_moons_classification_incorrect_noise_std(noise_std: float | int) -> None:
    with raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        make_moons_classification(noise_std=noise_std)


@mark.parametrize("ratio", (-0.1, 1.0, 2.0))
def test_make_moons_classification_incorrect_ratio(ratio: float | int) -> None:
    with raises(
        RuntimeError,
        match=r"Incorrect value for ratio. Expected a value in interval \[0.0, 1.0\)",
    ):
        make_moons_classification(ratio=ratio)


def test_make_moons_classification() -> None:
    data = make_moons_classification(num_examples=100)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    targets = data[ct.TARGET]
    assert isinstance(targets, BatchedTensor)
    assert targets.batch_size == 100
    assert targets.shape == (100,)
    assert targets.dtype == torch.long
    assert targets.sum() == 50
    assert targets.min() >= 0
    assert targets.max() <= 1

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert features.batch_size == 100
    assert features.shape == (100, 2)
    assert features.dtype == torch.float
    assert features.min() >= -2.0
    assert features.max() <= 2.0


def test_make_moons_classification_shuffle_false() -> None:
    data = make_moons_classification(num_examples=10, shuffle=False)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert data[ct.TARGET].equal(BatchedTensor(torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])))

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert features.batch_size == 10
    assert features.shape == (10, 2)
    assert features.dtype == torch.float
    assert features.min() >= -2.0
    assert features.max() <= 2.0


def test_make_moons_classification_ratio_0_2() -> None:
    data = make_moons_classification(num_examples=10, shuffle=False, ratio=0.2)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert data[ct.TARGET].equal(BatchedTensor(torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])))

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert features.batch_size == 10
    assert features.shape == (10, 2)
    assert features.dtype == torch.float
    assert features.min() >= -2.0
    assert features.max() <= 2.0


@mark.parametrize("num_examples", SIZES)
def test_make_moons_classification_num_examples(num_examples: int) -> None:
    data = make_moons_classification(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


@mark.parametrize("noise_std", (0.0, 1.0))
@mark.parametrize("shuffle", (True, False))
def test_make_moons_classification_same_random_seed(noise_std: float, shuffle: bool) -> None:
    assert objects_are_equal(
        make_moons_classification(
            num_examples=64,
            noise_std=noise_std,
            shuffle=shuffle,
            generator=get_torch_generator(1),
        ),
        make_moons_classification(
            num_examples=64,
            noise_std=noise_std,
            shuffle=shuffle,
            generator=get_torch_generator(1),
        ),
    )


@mark.parametrize("shuffle,noise_std", ((True, 0.0), (True, 1.0), (False, 1.0)))
def test_make_moons_classification_different_random_seeds(shuffle: bool, noise_std: float) -> None:
    assert not objects_are_equal(
        make_moons_classification(
            num_examples=64,
            noise_std=noise_std,
            shuffle=shuffle,
            generator=get_torch_generator(1),
        ),
        make_moons_classification(
            num_examples=64,
            noise_std=noise_std,
            shuffle=shuffle,
            generator=get_torch_generator(2),
        ),
    )
