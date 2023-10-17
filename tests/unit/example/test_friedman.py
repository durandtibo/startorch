from __future__ import annotations

import math
from unittest.mock import patch

import torch
from coola import objects_are_equal
from pytest import mark, raises
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example import (
    Friedman1Regression,
    Friedman2Regression,
    Friedman3Regression,
    make_friedman1_regression,
    make_friedman2_regression,
    make_friedman3_regression,
)
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


#########################################################
#     Tests for Friedman1RegressionExampleGenerator     #
#########################################################


def test_friedman1_regression_str() -> None:
    assert str(Friedman1Regression()).startswith("Friedman1RegressionExampleGenerator(")


@mark.parametrize("feature_size", (5, 8, 10))
def test_friedman1_regression_feature_size(feature_size: int) -> None:
    assert Friedman1Regression(feature_size=feature_size).feature_size == feature_size


@mark.parametrize("feature_size", (4, 1, 0, -1))
def test_friedman1_regression_incorrect_feature_size(feature_size: int) -> None:
    with raises(ValueError, match="feature_size (.*) has to be greater or equal to 5"):
        Friedman1Regression(feature_size=feature_size)


@mark.parametrize("noise_std", (0, 0.1, 1))
def test_friedman1_regression_noise_std(noise_std: float) -> None:
    assert Friedman1Regression(noise_std=noise_std).noise_std == noise_std


def test_friedman1_regression_incorrect_noise_std() -> None:
    with raises(
        ValueError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        Friedman1Regression(noise_std=-1)


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", (5, 8, 10))
def test_friedman1_regression_generate(batch_size: int, feature_size: int) -> None:
    data = Friedman1Regression(feature_size=feature_size).generate(batch_size)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].shape == (batch_size, feature_size)
    assert data[ct.FEATURE].dtype == torch.float


@mark.parametrize("noise_std", (0.0, 1.0))
def test_friedman1_regression_generate_same_random_seed(noise_std: float) -> None:
    generator = Friedman1Regression(feature_size=8, noise_std=noise_std)
    assert generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1))
    )


@mark.parametrize("noise_std", (0.0, 1.0))
def test_friedman1_regression_generate_different_random_seeds(noise_std: float) -> None:
    generator = Friedman1Regression(feature_size=8, noise_std=noise_std)
    assert not generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(2))
    )


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("noise_std", (0.0, 1.0))
@mark.parametrize("feature_size", (5, 10))
@mark.parametrize("rng", (None, get_torch_generator(1)))
def test_friedman1_regression_generate_mock(
    batch_size: int,
    noise_std: float,
    feature_size: int,
    rng: torch.Generator | None,
) -> None:
    generator = Friedman1Regression(noise_std=noise_std, feature_size=feature_size)
    with patch("startorch.example.friedman.make_friedman1_regression") as make_mock:
        generator.generate(batch_size=batch_size, rng=rng)
        make_mock.assert_called_once_with(
            num_examples=batch_size,
            noise_std=noise_std,
            feature_size=feature_size,
            generator=rng,
        )


#########################################################
#     Tests for Friedman2RegressionExampleGenerator     #
#########################################################


def test_friedman2_regression_str() -> None:
    assert str(Friedman2Regression()).startswith("Friedman2RegressionExampleGenerator(")


@mark.parametrize("feature_size", (4, 8, 10))
def test_friedman2_regression_feature_size(feature_size: int) -> None:
    assert Friedman2Regression(feature_size=feature_size).feature_size == feature_size


@mark.parametrize("feature_size", (3, 1, 0, -1))
def test_friedman2_regression_incorrect_feature_size(feature_size: int) -> None:
    with raises(ValueError, match="feature_size (.*) has to be greater or equal to 4"):
        Friedman2Regression(feature_size=feature_size)


@mark.parametrize("noise_std", (0, 0.1, 1))
def test_friedman2_regression_noise_std(noise_std: float) -> None:
    assert Friedman2Regression(noise_std=noise_std).noise_std == noise_std


def test_friedman2_regression_incorrect_noise_std() -> None:
    with raises(
        ValueError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        Friedman2Regression(noise_std=-1)


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", (5, 8, 10))
def test_friedman2_regression_generate(batch_size: int, feature_size: int) -> None:
    data = Friedman2Regression(feature_size=feature_size).generate(batch_size)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].shape == (batch_size, feature_size)
    assert data[ct.FEATURE].dtype == torch.float


@mark.parametrize("noise_std", (0.0, 1.0))
def test_friedman2_regression_generate_same_random_seed(noise_std: float) -> None:
    generator = Friedman2Regression(feature_size=8, noise_std=noise_std)
    assert generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1))
    )


@mark.parametrize("noise_std", (0.0, 1.0))
def test_friedman2_regression_generate_different_random_seeds(noise_std: float) -> None:
    generator = Friedman2Regression(feature_size=8, noise_std=noise_std)
    assert not generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(2))
    )


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("noise_std", (0.0, 1.0))
@mark.parametrize("feature_size", (4, 8))
@mark.parametrize("rng", (None, get_torch_generator(1)))
def test_friedman2_regression_generate_mock(
    batch_size: int,
    noise_std: float,
    feature_size: int,
    rng: torch.Generator | None,
) -> None:
    generator = Friedman2Regression(noise_std=noise_std, feature_size=feature_size)
    with patch("startorch.example.friedman.make_friedman2_regression") as make_mock:
        generator.generate(batch_size=batch_size, rng=rng)
        make_mock.assert_called_once_with(
            num_examples=batch_size,
            noise_std=noise_std,
            feature_size=feature_size,
            generator=rng,
        )


#########################################################
#     Tests for Friedman3RegressionExampleGenerator     #
#########################################################


def test_friedman3_regression_str() -> None:
    assert str(Friedman3Regression()).startswith("Friedman3RegressionExampleGenerator(")


@mark.parametrize("feature_size", (4, 8, 10))
def test_friedman3_regression_feature_size(feature_size: int) -> None:
    assert Friedman3Regression(feature_size=feature_size).feature_size == feature_size


@mark.parametrize("feature_size", (3, 1, 0, -1))
def test_friedman3_regression_incorrect_feature_size(feature_size: int) -> None:
    with raises(ValueError, match="feature_size (.*) has to be greater or equal to 4"):
        Friedman3Regression(feature_size=feature_size)


@mark.parametrize("noise_std", (0, 0.1, 1))
def test_friedman3_regression_noise_std(noise_std: float) -> None:
    assert Friedman3Regression(noise_std=noise_std).noise_std == noise_std


def test_friedman3_regression_incorrect_noise_std() -> None:
    with raises(
        ValueError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        Friedman3Regression(noise_std=-1)


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", (5, 8, 10))
def test_friedman3_regression_generate(batch_size: int, feature_size: int) -> None:
    data = Friedman3Regression(feature_size=feature_size).generate(batch_size)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].shape == (batch_size, feature_size)
    assert data[ct.FEATURE].dtype == torch.float


@mark.parametrize("noise_std", (0.0, 1.0))
def test_friedman3_regression_generate_same_random_seed(noise_std: float) -> None:
    generator = Friedman3Regression(feature_size=8, noise_std=noise_std)
    assert generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1))
    )


@mark.parametrize("noise_std", (0.0, 1.0))
def test_friedman3_regression_generate_different_random_seeds(noise_std: float) -> None:
    generator = Friedman3Regression(feature_size=8, noise_std=noise_std)
    assert not generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(2))
    )


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("noise_std", (0.0, 1.0))
@mark.parametrize("feature_size", (4, 8))
@mark.parametrize("rng", (None, get_torch_generator(1)))
def test_friedman3_regression_generate_mock(
    batch_size: int,
    noise_std: float,
    feature_size: int,
    rng: torch.Generator | None,
) -> None:
    generator = Friedman3Regression(noise_std=noise_std, feature_size=feature_size)
    with patch("startorch.example.friedman.make_friedman3_regression") as make_mock:
        generator.generate(batch_size=batch_size, rng=rng)
        make_mock.assert_called_once_with(
            num_examples=batch_size,
            noise_std=noise_std,
            feature_size=feature_size,
            generator=rng,
        )


###############################################
#     Tests for make_friedman1_regression     #
###############################################


@mark.parametrize("num_examples", (0, -1))
def test_make_friedman1_regression_incorrect_num_examples(num_examples: int) -> None:
    with raises(RuntimeError, match="The number of examples .* has to be greater than 0"):
        make_friedman1_regression(num_examples=num_examples)


@mark.parametrize("feature_size", (4, 1, 0, -1))
def test_make_friedman1_regression_incorrect_feature_size(feature_size: int) -> None:
    with raises(RuntimeError, match="feature_size (.*) has to be greater or equal to 5"):
        make_friedman1_regression(feature_size=feature_size)


def test_make_friedman1_regression_incorrect_noise_std() -> None:
    with raises(
        RuntimeError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        make_friedman1_regression(noise_std=-1)


def test_make_friedman1_regression() -> None:
    data = make_friedman1_regression(num_examples=10, feature_size=8)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].shape == (10, 8)
    assert data[ct.FEATURE].dtype == torch.float


@mark.parametrize("num_examples", SIZES)
def test_make_friedman1_regression_num_examples(num_examples: int) -> None:
    data = make_friedman1_regression(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


@mark.parametrize("feature_size", (5, 8, 10))
def test_make_friedman1_regression_feature_size(feature_size: int) -> None:
    data = make_friedman1_regression(num_examples=10, feature_size=feature_size)
    assert data[ct.FEATURE].shape[1] == feature_size


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_friedman1_regression_same_random_seed(noise_std: float) -> None:
    assert objects_are_equal(
        make_friedman1_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_friedman1_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
    )


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_friedman1_regression_different_random_seeds(noise_std: float) -> None:
    assert not objects_are_equal(
        make_friedman1_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_friedman1_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(2),
        ),
    )


###############################################
#     Tests for make_friedman2_regression     #
###############################################


@mark.parametrize("num_examples", (0, -1))
def test_make_friedman2_regression_incorrect_num_examples(num_examples: int) -> None:
    with raises(RuntimeError, match="The number of examples .* has to be greater than 0"):
        make_friedman2_regression(num_examples=num_examples)


@mark.parametrize("feature_size", (3, 1, 0, -1))
def test_make_friedman2_regression_incorrect_feature_size(feature_size: int) -> None:
    with raises(RuntimeError, match="feature_size (.*) has to be greater or equal to 4"):
        make_friedman2_regression(feature_size=feature_size)


def test_make_friedman2_regression_incorrect_noise_std() -> None:
    with raises(
        RuntimeError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        make_friedman2_regression(noise_std=-1)


def test_make_friedman2_regression() -> None:
    data = make_friedman2_regression(num_examples=10)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    features = data[ct.FEATURE]
    assert isinstance(features, BatchedTensor)
    assert features.shape == (10, 4)
    assert features.dtype == torch.float

    assert torch.all(0.0 <= features[:, 0]) and torch.all(features[:, 0] <= 100.0)
    assert torch.all(40.0 * math.pi <= features[:, 1]) and torch.all(
        features[:, 1] <= 560.0 * math.pi
    )
    assert torch.all(0.0 <= features[:, 2]) and torch.all(features[:, 2] <= 1.0)
    assert torch.all(1.0 <= features[:, 3]) and torch.all(features[:, 3] <= 11.0)


def test_make_friedman2_regression_feature_size_8() -> None:
    data = make_friedman2_regression(num_examples=10, feature_size=8)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    features = data[ct.FEATURE]
    assert isinstance(features, BatchedTensor)
    assert features.shape == (10, 8)
    assert features.dtype == torch.float

    assert torch.all(0.0 <= features[:, 0]) and torch.all(features[:, 0] <= 100.0)
    assert torch.all(40.0 * math.pi <= features[:, 1]) and torch.all(
        features[:, 1] <= 560.0 * math.pi
    )
    assert torch.all(0.0 <= features[:, 2]) and torch.all(features[:, 2] <= 1.0)
    assert torch.all(1.0 <= features[:, 3]) and torch.all(features[:, 3] <= 11.0)
    assert torch.all(0.0 <= features[:, 4:]) and torch.all(features[:, 4:] <= 1)


@mark.parametrize("num_examples", SIZES)
def test_make_friedman2_regression_num_examples(num_examples: int) -> None:
    data = make_friedman2_regression(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


@mark.parametrize("feature_size", (5, 8, 10))
def test_make_friedman2_regression_feature_size(feature_size: int) -> None:
    data = make_friedman2_regression(num_examples=10, feature_size=feature_size)
    assert data[ct.FEATURE].shape[1] == feature_size


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_friedman2_regression_same_random_seed(noise_std: float) -> None:
    assert objects_are_equal(
        make_friedman2_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_friedman2_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
    )


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_friedman2_regression_different_random_seeds(noise_std: float) -> None:
    assert not objects_are_equal(
        make_friedman2_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_friedman2_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(2),
        ),
    )


###############################################
#     Tests for make_friedman3_regression     #
###############################################


@mark.parametrize("num_examples", (0, -1))
def test_make_friedman3_regression_incorrect_num_examples(num_examples: int) -> None:
    with raises(RuntimeError, match="The number of examples .* has to be greater than 0"):
        make_friedman3_regression(num_examples=num_examples)


@mark.parametrize("feature_size", (3, 1, 0, -1))
def test_make_friedman3_regression_incorrect_feature_size(feature_size: int) -> None:
    with raises(RuntimeError, match="feature_size (.*) has to be greater or equal to 4"):
        make_friedman3_regression(feature_size=feature_size)


def test_make_friedman3_regression_incorrect_noise_std() -> None:
    with raises(
        RuntimeError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        make_friedman3_regression(noise_std=-1)


def test_make_friedman3_regression() -> None:
    data = make_friedman3_regression(num_examples=10)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    features = data[ct.FEATURE]
    assert isinstance(features, BatchedTensor)
    assert features.shape == (10, 4)
    assert features.dtype == torch.float

    assert torch.all(0.0 <= features[:, 0]) and torch.all(features[:, 0] <= 100.0)
    assert torch.all(40.0 * math.pi <= features[:, 1]) and torch.all(
        features[:, 1] <= 560.0 * math.pi
    )
    assert torch.all(0.0 <= features[:, 2]) and torch.all(features[:, 2] <= 1.0)
    assert torch.all(1.0 <= features[:, 3]) and torch.all(features[:, 3] <= 11.0)


def test_make_friedman3_regression_feature_size_8() -> None:
    data = make_friedman3_regression(num_examples=10, feature_size=8)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    features = data[ct.FEATURE]
    assert isinstance(features, BatchedTensor)
    assert features.shape == (10, 8)
    assert features.dtype == torch.float

    assert torch.all(0.0 <= features[:, 0]) and torch.all(features[:, 0] <= 100.0)
    assert torch.all(40.0 * math.pi <= features[:, 1]) and torch.all(
        features[:, 1] <= 560.0 * math.pi
    )
    assert torch.all(0.0 <= features[:, 2]) and torch.all(features[:, 2] <= 1.0)
    assert torch.all(1.0 <= features[:, 3]) and torch.all(features[:, 3] <= 11.0)
    assert torch.all(0.0 <= features[:, 4:]) and torch.all(features[:, 4:] <= 1)


@mark.parametrize("num_examples", SIZES)
def test_make_friedman3_regression_num_examples(num_examples: int) -> None:
    data = make_friedman3_regression(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


@mark.parametrize("feature_size", (5, 8, 10))
def test_make_friedman3_regression_feature_size(feature_size: int) -> None:
    data = make_friedman3_regression(num_examples=10, feature_size=feature_size)
    assert data[ct.FEATURE].shape[1] == feature_size


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_friedman3_regression_same_random_seed(noise_std: float) -> None:
    assert objects_are_equal(
        make_friedman3_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_friedman3_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
    )


@mark.parametrize("noise_std", (0.0, 1.0))
def test_make_friedman3_regression_different_random_seeds(noise_std: float) -> None:
    assert not objects_are_equal(
        make_friedman3_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(1),
        ),
        make_friedman3_regression(
            num_examples=10,
            feature_size=8,
            noise_std=noise_std,
            generator=get_torch_generator(2),
        ),
    )
