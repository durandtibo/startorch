from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch
from coola import objects_are_equal

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

SIZES = [1, 2, 4]


#########################################################
#     Tests for Friedman1RegressionExampleGenerator     #
#########################################################


def test_friedman1_regression_str() -> None:
    assert str(Friedman1Regression()).startswith("Friedman1RegressionExampleGenerator(")


@pytest.mark.parametrize("feature_size", [5, 8, 10])
def test_friedman1_regression_feature_size(feature_size: int) -> None:
    assert Friedman1Regression(feature_size=feature_size).feature_size == feature_size


@pytest.mark.parametrize("feature_size", [4, 1, 0, -1])
def test_friedman1_regression_incorrect_feature_size(feature_size: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for feature_size. Expected a value greater or equal to 5",
    ):
        Friedman1Regression(feature_size=feature_size)


@pytest.mark.parametrize("noise_std", [0, 0.1, 1])
def test_friedman1_regression_noise_std(noise_std: float) -> None:
    assert Friedman1Regression(noise_std=noise_std).noise_std == noise_std


@pytest.mark.parametrize("noise_std", [-1, -4.2])
def test_friedman1_regression_incorrect_noise_std(noise_std: float) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        Friedman1Regression(noise_std=noise_std)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", [5, 8, 10])
def test_friedman1_regression_generate(batch_size: int, feature_size: int) -> None:
    data = Friedman1Regression(feature_size=feature_size).generate(batch_size)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (batch_size, feature_size)
    assert data[ct.FEATURE].dtype == torch.float


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_friedman1_regression_generate_same_random_seed(noise_std: float) -> None:
    generator = Friedman1Regression(feature_size=8, noise_std=noise_std)
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_friedman1_regression_generate_different_random_seeds(noise_std: float) -> None:
    generator = Friedman1Regression(feature_size=8, noise_std=noise_std)
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("feature_size", [5, 10])
@pytest.mark.parametrize("rng", [None, get_torch_generator(1)])
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


@pytest.mark.parametrize("feature_size", [4, 8, 10])
def test_friedman2_regression_feature_size(feature_size: int) -> None:
    assert Friedman2Regression(feature_size=feature_size).feature_size == feature_size


@pytest.mark.parametrize("feature_size", [3, 1, 0, -1])
def test_friedman2_regression_incorrect_feature_size(feature_size: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for feature_size. Expected a value greater or equal to 4",
    ):
        Friedman2Regression(feature_size=feature_size)


@pytest.mark.parametrize("noise_std", [0, 0.1, 1])
def test_friedman2_regression_noise_std(noise_std: float) -> None:
    assert Friedman2Regression(noise_std=noise_std).noise_std == noise_std


@pytest.mark.parametrize("noise_std", [-1, -4.2])
def test_friedman2_regression_incorrect_noise_std(noise_std: float) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        Friedman2Regression(noise_std=noise_std)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", [5, 8, 10])
def test_friedman2_regression_generate(batch_size: int, feature_size: int) -> None:
    data = Friedman2Regression(feature_size=feature_size).generate(batch_size)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (batch_size, feature_size)
    assert data[ct.FEATURE].dtype == torch.float


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_friedman2_regression_generate_same_random_seed(noise_std: float) -> None:
    generator = Friedman2Regression(feature_size=8, noise_std=noise_std)
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_friedman2_regression_generate_different_random_seeds(noise_std: float) -> None:
    generator = Friedman2Regression(feature_size=8, noise_std=noise_std)
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("feature_size", [4, 8])
@pytest.mark.parametrize("rng", [None, get_torch_generator(1)])
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


@pytest.mark.parametrize("feature_size", [4, 8, 10])
def test_friedman3_regression_feature_size(feature_size: int) -> None:
    assert Friedman3Regression(feature_size=feature_size).feature_size == feature_size


@pytest.mark.parametrize("feature_size", [3, 1, 0, -1])
def test_friedman3_regression_incorrect_feature_size(feature_size: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for feature_size. Expected a value greater or equal to 4",
    ):
        Friedman3Regression(feature_size=feature_size)


@pytest.mark.parametrize("noise_std", [0, 0.1, 1])
def test_friedman3_regression_noise_std(noise_std: float) -> None:
    assert Friedman3Regression(noise_std=noise_std).noise_std == noise_std


@pytest.mark.parametrize("noise_std", [-1, -4.2])
def test_friedman3_regression_incorrect_noise_std(noise_std: float) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        Friedman3Regression(noise_std=noise_std)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", [5, 8, 10])
def test_friedman3_regression_generate(batch_size: int, feature_size: int) -> None:
    data = Friedman3Regression(feature_size=feature_size).generate(batch_size)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (batch_size, feature_size)
    assert data[ct.FEATURE].dtype == torch.float


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_friedman3_regression_generate_same_random_seed(noise_std: float) -> None:
    generator = Friedman3Regression(feature_size=8, noise_std=noise_std)
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_friedman3_regression_generate_different_random_seeds(noise_std: float) -> None:
    generator = Friedman3Regression(feature_size=8, noise_std=noise_std)
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("feature_size", [4, 8])
@pytest.mark.parametrize("rng", [None, get_torch_generator(1)])
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


@pytest.mark.parametrize("num_examples", [0, -1])
def test_make_friedman1_regression_incorrect_num_examples(num_examples: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for num_examples. Expected a value greater or equal to 1",
    ):
        make_friedman1_regression(num_examples=num_examples)


@pytest.mark.parametrize("feature_size", [4, 1, 0, -1])
def test_make_friedman1_regression_incorrect_feature_size(feature_size: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for feature_size. Expected a value greater or equal to 5",
    ):
        make_friedman1_regression(feature_size=feature_size)


@pytest.mark.parametrize("noise_std", [-1, -4.2])
def test_make_friedman1_regression_incorrect_noise_std(noise_std: float) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        make_friedman1_regression(noise_std=noise_std)


def test_make_friedman1_regression() -> None:
    data = make_friedman1_regression(num_examples=10, feature_size=8)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (10, 8)
    assert data[ct.FEATURE].dtype == torch.float


@pytest.mark.parametrize("num_examples", SIZES)
def test_make_friedman1_regression_num_examples(num_examples: int) -> None:
    data = make_friedman1_regression(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].shape[0] == num_examples
    assert data[ct.FEATURE].shape[0] == num_examples


@pytest.mark.parametrize("feature_size", [5, 8, 10])
def test_make_friedman1_regression_feature_size(feature_size: int) -> None:
    data = make_friedman1_regression(num_examples=10, feature_size=feature_size)
    assert data[ct.FEATURE].shape[1] == feature_size


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
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


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
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


@pytest.mark.parametrize("num_examples", [0, -1])
def test_make_friedman2_regression_incorrect_num_examples(num_examples: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for num_examples. Expected a value greater or equal to 1",
    ):
        make_friedman2_regression(num_examples=num_examples)


@pytest.mark.parametrize("feature_size", [3, 1, 0, -1])
def test_make_friedman2_regression_incorrect_feature_size(feature_size: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for feature_size. Expected a value greater or equal to 4",
    ):
        make_friedman2_regression(feature_size=feature_size)


@pytest.mark.parametrize("noise_std", [-1, -4.2])
def test_make_friedman2_regression_incorrect_noise_std(noise_std: float) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        make_friedman2_regression(noise_std=noise_std)


def test_make_friedman2_regression() -> None:
    data = make_friedman2_regression(num_examples=10)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    features = data[ct.FEATURE]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (10, 4)
    assert features.dtype == torch.float

    assert torch.all(features[:, 0] >= 0.0)
    assert torch.all(features[:, 0] <= 100.0)
    assert torch.all(40.0 * math.pi <= features[:, 1])
    assert torch.all(features[:, 1] <= 560.0 * math.pi)
    assert torch.all(features[:, 2] >= 0.0)
    assert torch.all(features[:, 2] <= 1.0)
    assert torch.all(features[:, 3] >= 1.0)
    assert torch.all(features[:, 3] <= 11.0)


def test_make_friedman2_regression_feature_size_8() -> None:
    data = make_friedman2_regression(num_examples=10, feature_size=8)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    features = data[ct.FEATURE]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (10, 8)
    assert features.dtype == torch.float

    assert torch.all(features[:, 0] >= 0.0)
    assert torch.all(features[:, 0] <= 100.0)
    assert torch.all(40.0 * math.pi <= features[:, 1])
    assert torch.all(features[:, 1] <= 560.0 * math.pi)
    assert torch.all(features[:, 2] >= 0.0)
    assert torch.all(features[:, 2] <= 1.0)
    assert torch.all(features[:, 3] >= 1.0)
    assert torch.all(features[:, 3] <= 11.0)
    assert torch.all(features[:, 4:] >= 0.0)
    assert torch.all(features[:, 4:] <= 1)


@pytest.mark.parametrize("num_examples", SIZES)
def test_make_friedman2_regression_num_examples(num_examples: int) -> None:
    data = make_friedman2_regression(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].shape[0] == num_examples
    assert data[ct.FEATURE].shape[0] == num_examples


@pytest.mark.parametrize("feature_size", [5, 8, 10])
def test_make_friedman2_regression_feature_size(feature_size: int) -> None:
    data = make_friedman2_regression(num_examples=10, feature_size=feature_size)
    assert data[ct.FEATURE].shape[1] == feature_size


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
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


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
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


@pytest.mark.parametrize("num_examples", [0, -1])
def test_make_friedman3_regression_incorrect_num_examples(num_examples: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for num_examples. Expected a value greater or equal to 1",
    ):
        make_friedman3_regression(num_examples=num_examples)


@pytest.mark.parametrize("feature_size", [3, 1, 0, -1])
def test_make_friedman3_regression_incorrect_feature_size(feature_size: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for feature_size. Expected a value greater or equal to 4",
    ):
        make_friedman3_regression(feature_size=feature_size)


@pytest.mark.parametrize("noise_std", [-1, -4.2])
def test_make_friedman3_regression_incorrect_noise_std(noise_std: float) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        make_friedman3_regression(noise_std=noise_std)


def test_make_friedman3_regression() -> None:
    data = make_friedman3_regression(num_examples=10)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    features = data[ct.FEATURE]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (10, 4)
    assert features.dtype == torch.float

    assert torch.all(features[:, 0] >= 0.0)
    assert torch.all(features[:, 0] <= 100.0)
    assert torch.all(40.0 * math.pi <= features[:, 1])
    assert torch.all(features[:, 1] <= 560.0 * math.pi)
    assert torch.all(features[:, 2] >= 0.0)
    assert torch.all(features[:, 2] <= 1.0)
    assert torch.all(features[:, 3] >= 1.0)
    assert torch.all(features[:, 3] <= 11.0)


def test_make_friedman3_regression_feature_size_8() -> None:
    data = make_friedman3_regression(num_examples=10, feature_size=8)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    features = data[ct.FEATURE]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (10, 8)
    assert features.dtype == torch.float

    assert torch.all(features[:, 0] >= 0.0)
    assert torch.all(features[:, 0] <= 100.0)
    assert torch.all(40.0 * math.pi <= features[:, 1])
    assert torch.all(features[:, 1] <= 560.0 * math.pi)
    assert torch.all(features[:, 2] >= 0.0)
    assert torch.all(features[:, 2] <= 1.0)
    assert torch.all(features[:, 3] >= 1.0)
    assert torch.all(features[:, 3] <= 11.0)
    assert torch.all(features[:, 4:] >= 0.0)
    assert torch.all(features[:, 4:] <= 1)


@pytest.mark.parametrize("num_examples", SIZES)
def test_make_friedman3_regression_num_examples(num_examples: int) -> None:
    data = make_friedman3_regression(num_examples)
    assert len(data) == 2
    assert data[ct.TARGET].shape[0] == num_examples
    assert data[ct.FEATURE].shape[0] == num_examples


@pytest.mark.parametrize("feature_size", [5, 8, 10])
def test_make_friedman3_regression_feature_size(feature_size: int) -> None:
    data = make_friedman3_regression(num_examples=10, feature_size=feature_size)
    assert data[ct.FEATURE].shape[1] == feature_size


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
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


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
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
