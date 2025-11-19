from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch.example import SwissRoll, make_swiss_roll
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


###############################################
#     Tests for SwissRollExampleGenerator     #
###############################################


def test_swiss_roll_str() -> None:
    assert str(SwissRoll()).startswith("SwissRollExampleGenerator(")


@pytest.mark.parametrize("noise_std", [0, 0.1, 1])
def test_swiss_roll_noise_std(noise_std: float) -> None:
    assert SwissRoll(noise_std=noise_std).noise_std == noise_std


def test_swiss_roll_incorrect_noise_std() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"Incorrect value for noise_std. Expected a value greater than 0",
    ):
        SwissRoll(noise_std=-1)


@pytest.mark.parametrize("spin", [0.1, 1, 2.0])
def test_swiss_roll_spin(spin: float) -> None:
    assert SwissRoll(spin=spin).spin == spin


@pytest.mark.parametrize("spin", [0, -0.1, -1.0])
def test_swiss_roll_incorrect_spin(spin: float) -> None:
    with pytest.raises(
        RuntimeError, match=r"Incorrect value for spin. Expected a value in interval"
    ):
        SwissRoll(spin=spin)


@pytest.mark.parametrize("batch_size", SIZES)
def test_swiss_roll_generate(batch_size: int) -> None:
    data = SwissRoll().generate(batch_size)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (batch_size, 3)
    assert data[ct.FEATURE].dtype == torch.float


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("hole", [True, False])
def test_swiss_roll_generate_same_random_seed(noise_std: float, hole: bool) -> None:
    generator = SwissRoll(noise_std=noise_std, hole=hole)
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("hole", [True, False])
def test_swiss_roll_generate_different_random_seeds(noise_std: float, hole: bool) -> None:
    generator = SwissRoll(noise_std=noise_std, hole=hole)
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )


def test_swiss_roll_generate_same_random_seed_hole() -> None:
    generator1 = SwissRoll(hole=True)
    generator2 = SwissRoll(hole=False)
    assert not objects_are_equal(
        generator1.generate(batch_size=64, rng=get_torch_generator(1)),
        generator2.generate(batch_size=64, rng=get_torch_generator(1)),
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("spin", [1.5, 2.0])
@pytest.mark.parametrize("hole", [True, False])
@pytest.mark.parametrize("rng", [None, get_torch_generator(1)])
def test_swiss_roll_generate_mock(
    batch_size: int, noise_std: float, spin: float, hole: bool, rng: torch.Generator | None
) -> None:
    generator = SwissRoll(noise_std=noise_std, hole=hole, spin=spin)
    with patch("startorch.example.swissroll.make_swiss_roll") as make_mock:
        generator.generate(batch_size=batch_size, rng=rng)
        make_mock.assert_called_once_with(
            num_examples=batch_size,
            noise_std=noise_std,
            spin=spin,
            hole=hole,
            generator=rng,
        )


#####################################
#     Tests for make_swiss_roll     #
#####################################


@pytest.mark.parametrize("num_examples", [0, -1])
def test_make_swiss_roll_incorrect_num_examples(num_examples: int) -> None:
    with pytest.raises(
        RuntimeError,
        match=r"Incorrect value for num_examples. Expected a value greater or equal to 1",
    ):
        make_swiss_roll(num_examples=num_examples)


def test_make_swiss_roll_incorrect_noise_std() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"Incorrect value for noise_std. Expected a value greater than 0",
    ):
        make_swiss_roll(noise_std=-1)


@pytest.mark.parametrize("spin", [0, -1])
def test_make_swiss_roll_incorrect_spin(spin: int) -> None:
    with pytest.raises(
        RuntimeError, match=r"Incorrect value for spin. Expected a value in interval"
    ):
        make_swiss_roll(num_examples=10, spin=spin)


def test_make_swiss_roll() -> None:
    data = make_swiss_roll(num_examples=64)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (64,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (64, 3)
    assert data[ct.FEATURE].dtype == torch.float


@pytest.mark.parametrize("spin", [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("noise_std", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("hole", [True, False])
def test_make_swiss_roll_params(spin: float, noise_std: float, hole: bool) -> None:
    data = make_swiss_roll(num_examples=64, spin=spin, noise_std=noise_std, hole=hole)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (64,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (64, 3)
    assert data[ct.FEATURE].dtype == torch.float


@pytest.mark.parametrize("num_examples", SIZES)
def test_make_swiss_roll_num_examples(num_examples: int) -> None:
    data = make_swiss_roll(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].shape[0] == num_examples
    assert data[ct.FEATURE].shape[0] == num_examples


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("hole", [True, False])
def test_make_swiss_roll_create_same_random_seed(noise_std: float, hole: bool) -> None:
    assert objects_are_equal(
        make_swiss_roll(
            num_examples=64, noise_std=noise_std, hole=hole, generator=get_torch_generator(1)
        ),
        make_swiss_roll(
            num_examples=64, noise_std=noise_std, hole=hole, generator=get_torch_generator(1)
        ),
    )


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("hole", [True, False])
def test_make_swiss_roll_create_different_random_seeds(noise_std: float, hole: bool) -> None:
    assert not objects_are_equal(
        make_swiss_roll(
            num_examples=64, noise_std=noise_std, hole=hole, generator=get_torch_generator(1)
        ),
        make_swiss_roll(
            num_examples=64, noise_std=noise_std, hole=hole, generator=get_torch_generator(2)
        ),
    )


def test_make_swiss_roll_create_same_random_seed_hole() -> None:
    assert not objects_are_equal(
        make_swiss_roll(num_examples=64, hole=True, generator=get_torch_generator(1)),
        make_swiss_roll(num_examples=64, hole=False, generator=get_torch_generator(1)),
    )
