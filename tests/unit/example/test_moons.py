from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch.example import MoonsClassification, make_moons_classification
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


#########################################################
#     Tests for MoonsClassificationExampleGenerator     #
#########################################################


def test_moons_classification_str() -> None:
    assert str(MoonsClassification()).startswith("MoonsClassificationExampleGenerator(")


@pytest.mark.parametrize("noise_std", [0, 0.1, 1])
def test_moons_classification_noise_std(noise_std: float) -> None:
    assert MoonsClassification(noise_std=noise_std).noise_std == noise_std


def test_moons_classification_incorrect_noise_std() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"Incorrect value for noise_std. Expected a value greater than 0",
    ):
        MoonsClassification(noise_std=-1)


@pytest.mark.parametrize("ratio", [0.5, 0.8])
def test_moons_classification_ratio(ratio: float) -> None:
    assert MoonsClassification(ratio=ratio).ratio == ratio


@pytest.mark.parametrize("ratio", [-0.1, 1.0])
def test_moons_classification_incorrect_ratio(ratio: float) -> None:
    with pytest.raises(
        RuntimeError, match=r"Incorrect value for ratio. Expected a value in interval"
    ):
        MoonsClassification(ratio=ratio)


@pytest.mark.parametrize("batch_size", SIZES)
def test_moons_classification_generate(batch_size: int) -> None:
    data = MoonsClassification().generate(batch_size)
    assert isinstance(data, dict)
    assert len(data) == 2
    targets = data[ct.TARGET]
    assert isinstance(targets, torch.Tensor)
    assert targets.shape == (batch_size,)
    assert targets.dtype == torch.long
    assert targets.min() >= 0
    assert targets.max() <= 1

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert features.shape == (batch_size, 2)
    assert features.dtype == torch.float
    assert features.min() >= -1.0
    assert features.max() <= 2.0


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("shuffle", [True, False])
def test_moons_classification_generate_same_random_seed(noise_std: float, shuffle: bool) -> None:
    generator = MoonsClassification(noise_std=noise_std, shuffle=shuffle)
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_moons_classification_generate_different_random_seeds(noise_std: float) -> None:
    generator = MoonsClassification(noise_std=noise_std)
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )


@pytest.mark.parametrize(
    ("batch_size", "noise_std", "ratio", "shuffle", "rng"),
    [
        (2, 0.0, 0.5, True, None),
        (4, 0.5, 0.2, False, get_torch_generator(1)),
    ],
)
def test_moons_classification_generate_mock(
    batch_size: int,
    noise_std: float,
    ratio: float,
    shuffle: bool,
    rng: torch.Generator | None,
) -> None:
    generator = MoonsClassification(noise_std=noise_std, shuffle=shuffle, ratio=ratio)
    with patch("startorch.example.moons.make_moons_classification") as make_mock:
        generator.generate(batch_size=batch_size, rng=rng)
        make_mock.assert_called_once_with(
            num_examples=batch_size,
            shuffle=shuffle,
            noise_std=noise_std,
            ratio=ratio,
            generator=rng,
        )


###############################################
#     Tests for make_moons_classification     #
###############################################


@pytest.mark.parametrize("num_examples", [0, -1])
def test_make_moons_classification_incorrect_num_examples(num_examples: int) -> None:
    with pytest.raises(
        RuntimeError,
        match=r"Incorrect value for num_examples. Expected a value greater or equal to 1",
    ):
        make_moons_classification(num_examples=num_examples)


@pytest.mark.parametrize("noise_std", [-1, -4.2])
def test_make_moons_classification_incorrect_noise_std(noise_std: float) -> None:
    with pytest.raises(
        RuntimeError,
        match=r"Incorrect value for noise_std. Expected a value greater than 0",
    ):
        make_moons_classification(noise_std=noise_std)


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 2.0])
def test_make_moons_classification_incorrect_ratio(ratio: float) -> None:
    with pytest.raises(
        RuntimeError,
        match=r"Incorrect value for ratio. Expected a value in interval \[0.0, 1.0\)",
    ):
        make_moons_classification(ratio=ratio)


def test_make_moons_classification() -> None:
    data = make_moons_classification(num_examples=100)
    assert isinstance(data, dict)
    assert len(data) == 2
    targets = data[ct.TARGET]
    assert isinstance(targets, torch.Tensor)
    assert targets.shape == (100,)
    assert targets.dtype == torch.long
    assert targets.sum() == 50
    assert targets.min() >= 0
    assert targets.max() <= 1

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert features.shape == (100, 2)
    assert features.dtype == torch.float
    assert features.min() >= -1.0
    assert features.max() <= 2.0


def test_make_moons_classification_shuffle_false() -> None:
    data = make_moons_classification(num_examples=10, shuffle=False)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert objects_are_equal(
        data[ct.TARGET], torch.Tensor(torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
    )

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert features.shape == (10, 2)
    assert features.dtype == torch.float
    assert features.min() >= -1.0
    assert features.max() <= 2.0


def test_make_moons_classification_ratio_0_2() -> None:
    data = make_moons_classification(num_examples=10, shuffle=False, ratio=0.2)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert objects_are_equal(data[ct.TARGET], torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert features.shape == (10, 2)
    assert features.dtype == torch.float
    assert features.min() >= -1.0
    assert features.max() <= 2.0


@pytest.mark.parametrize("num_examples", SIZES)
def test_make_moons_classification_num_examples(num_examples: int) -> None:
    data = make_moons_classification(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].shape[0] == num_examples
    assert data[ct.FEATURE].shape[0] == num_examples


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("shuffle", [True, False])
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


@pytest.mark.parametrize(("shuffle", "noise_std"), [(True, 0.0), (True, 1.0), (False, 1.0)])
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
