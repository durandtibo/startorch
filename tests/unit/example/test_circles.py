from __future__ import annotations

from unittest.mock import patch

import torch
from coola import objects_are_equal
from pytest import mark, raises
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example import CirclesClassification, make_circles_classification
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


###########################################################
#     Tests for CirclesClassificationExampleGenerator     #
###########################################################


def test_circles_classification_str() -> None:
    assert str(CirclesClassification()).startswith("CirclesClassificationExampleGenerator(")


@mark.parametrize("noise_std", (0, 0.1, 1))
def test_circles_classification_noise_std(noise_std: float) -> None:
    assert CirclesClassification(noise_std=noise_std).noise_std == noise_std


def test_circles_classification_incorrect_noise_std() -> None:
    with raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        CirclesClassification(noise_std=-1)


@mark.parametrize("factor", (0.5, 0.8))
def test_circles_classification_factor(factor: float) -> None:
    assert CirclesClassification(factor=factor).factor == factor


@mark.parametrize("factor", (-0.1, 1.0))
def test_circles_classification_incorrect_factor(factor: float) -> None:
    with raises(RuntimeError, match="Incorrect value for factor. Expected a value in interval"):
        CirclesClassification(factor=factor)


@mark.parametrize("ratio", (0.5, 0.8))
def test_circles_classification_ratio(ratio: float) -> None:
    assert CirclesClassification(ratio=ratio).ratio == ratio


@mark.parametrize("ratio", (-0.1, 1.0))
def test_circles_classification_incorrect_ratio(ratio: float) -> None:
    with raises(RuntimeError, match="Incorrect value for ratio. Expected a value in interval"):
        CirclesClassification(ratio=ratio)


@mark.parametrize("batch_size", SIZES)
def test_circles_classification_generate(batch_size: int) -> None:
    data = CirclesClassification().generate(batch_size)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].batch_size == batch_size
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.long
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].batch_size == batch_size
    assert data[ct.FEATURE].shape == (batch_size, 2)
    assert data[ct.FEATURE].dtype == torch.float


@mark.parametrize("noise_std", (0.0, 1.0))
@mark.parametrize("shuffle", (True, False))
def test_circles_classification_generate_same_random_seed(
    noise_std: float | int, shuffle: bool
) -> None:
    generator = CirclesClassification(noise_std=noise_std, shuffle=shuffle)
    assert generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1))
    )


@mark.parametrize("noise_std", (0.0, 1.0))
def test_circles_classification_generate_different_random_seeds(noise_std: float | int) -> None:
    generator = CirclesClassification(noise_std=noise_std)
    assert not generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(2))
    )


@mark.parametrize(
    "batch_size,noise_std,ratio,factor,shuffle,rng",
    (
        (2, 0.0, 0.5, 0.8, True, None),
        (4, 0.5, 0.2, 0.5, False, get_torch_generator(1)),
    ),
)
def test_circles_classification_generate_mock(
    batch_size: int,
    noise_std: float | int,
    factor: float,
    ratio: float,
    shuffle: bool,
    rng: torch.Generator | None,
) -> None:
    generator = CirclesClassification(
        noise_std=noise_std, shuffle=shuffle, ratio=ratio, factor=factor
    )
    with patch("startorch.example.circles.make_circles_classification") as make_mock:
        generator.generate(batch_size=batch_size, rng=rng)
        make_mock.assert_called_once_with(
            num_examples=batch_size,
            shuffle=shuffle,
            noise_std=noise_std,
            factor=factor,
            ratio=ratio,
            generator=rng,
        )


#################################################
#     Tests for make_circles_classification     #
#################################################


@mark.parametrize("num_examples", (0, -1))
def test_make_circles_classification_incorrect_num_examples(num_examples: int) -> None:
    with raises(
        RuntimeError,
        match="Incorrect value for num_examples. Expected a value greater or equal to 1",
    ):
        make_circles_classification(num_examples=num_examples)


@mark.parametrize("noise_std", (-1, -4.2))
def test_make_circles_classification_incorrect_noise_std(noise_std: float | int) -> None:
    with raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        make_circles_classification(noise_std=noise_std)


@mark.parametrize("factor", (-0.1, 1.0, 2.0))
def test_make_circles_classification_incorrect_factor(factor: float | int) -> None:
    with raises(
        RuntimeError,
        match=r"Incorrect value for factor. Expected a value in interval \[0.0, 1.0\)",
    ):
        make_circles_classification(factor=factor)


@mark.parametrize("ratio", (-0.1, 1.0, 2.0))
def test_make_circles_classification_incorrect_ratio(ratio: float | int) -> None:
    with raises(
        RuntimeError,
        match=r"Incorrect value for ratio. Expected a value in interval \[0.0, 1.0\)",
    ):
        make_circles_classification(ratio=ratio)


def test_make_circles_classification() -> None:
    data = make_circles_classification(num_examples=100)
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
    assert features.min() >= -1.0
    assert features.max() <= 1.0


def test_make_circles_classification_shuffle_false() -> None:
    data = make_circles_classification(num_examples=10, shuffle=False)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert data[ct.TARGET].equal(BatchedTensor(torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])))

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert features.batch_size == 10
    assert features.shape == (10, 2)
    assert features.dtype == torch.float
    assert features.min() >= -1.0
    assert features.max() <= 1.0


@mark.parametrize("factor", (0.2, 0.5, 0.8))
def test_make_circles_classification_factor(factor: float) -> None:
    data = make_circles_classification(num_examples=100, factor=factor)
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
    assert features.min() >= -1.0
    assert features.max() <= 1.0


@mark.parametrize("num_examples", SIZES)
def test_make_circles_classification_num_examples(num_examples: int) -> None:
    data = make_circles_classification(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_circles_classification_same_random_seed(noise_std: float) -> None:
    assert objects_are_equal(
        make_circles_classification(
            num_examples=64,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_circles_classification(
            num_examples=64,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
    )


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_circles_classification_different_random_seeds(noise_std: float) -> None:
    assert not objects_are_equal(
        make_circles_classification(
            num_examples=64,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_circles_classification(
            num_examples=64,
            noise_std=noise_std,
            generator=get_torch_generator(2),
        ),
    )
