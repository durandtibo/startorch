from __future__ import annotations

import torch
from coola import objects_are_equal
from pytest import mark, raises
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example.swissroll import make_swiss_roll
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


#####################################
#     Tests for make_swiss_roll     #
#####################################


@mark.parametrize("num_examples", (0, -1))
def test_make_swiss_roll_incorrect_num_examples(num_examples: int) -> None:
    with raises(RuntimeError, match="The number of examples .* has to be greater than 0"):
        make_swiss_roll(num_examples=num_examples)


def test_make_swiss_roll_incorrect_noise_std() -> None:
    with raises(
        RuntimeError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        make_swiss_roll(noise_std=-1)


@mark.parametrize("spin", (0, -1))
def test_make_swiss_roll_incorrect_spin(spin: int) -> None:
    with raises(
        RuntimeError, match="The spin of the Swiss roll (.*) has to be greater or equal than 0"
    ):
        make_swiss_roll(num_examples=10, spin=spin)


def test_make_swiss_roll() -> None:
    data = make_swiss_roll(num_examples=100)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (100,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].shape == (100, 3)
    assert data[ct.FEATURE].dtype == torch.float


@mark.parametrize("spin", (0.5, 1.0, 1.5, 2.0))
@mark.parametrize("noise_std", (0.0, 0.5, 1.0))
def test_make_swiss_roll_params(spin: float | int, noise_std: float | int) -> None:
    data = make_swiss_roll(num_examples=100, spin=spin, noise_std=noise_std)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (100,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].shape == (100, 3)
    assert data[ct.FEATURE].dtype == torch.float


@mark.parametrize("num_examples", SIZES)
def test_make_swiss_roll_num_examples(num_examples: int) -> None:
    data = make_swiss_roll(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


def test_make_swiss_roll_create_same_random_seed() -> None:
    assert objects_are_equal(
        make_swiss_roll(num_examples=100, noise_std=1.0, generator=get_torch_generator(1)),
        make_swiss_roll(num_examples=100, noise_std=1.0, generator=get_torch_generator(1)),
    )


def test_make_swiss_roll_create_different_random_seeds() -> None:
    assert not objects_are_equal(
        make_swiss_roll(num_examples=100, noise_std=1.0, generator=get_torch_generator(1)),
        make_swiss_roll(num_examples=100, noise_std=1.0, generator=get_torch_generator(2)),
    )
