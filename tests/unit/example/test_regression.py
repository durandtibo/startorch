from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch.example import LinearRegression, make_linear_regression
from startorch.example.regression import get_uniform_weights
from startorch.utils.seed import get_torch_generator

if TYPE_CHECKING:
    from collections.abc import Sequence

SIZES = [1, 2, 4]


######################################################
#     Tests for LinearRegressionExampleGenerator     #
######################################################


def test_linear_regression_str() -> None:
    assert str(LinearRegression.create_uniform_weights()).startswith(
        "LinearRegressionExampleGenerator("
    )


@pytest.mark.parametrize(
    "weights", [torch.tensor([2.0, 1.0, 3.0]), [2.0, 1.0, 3.0], (2.0, 1.0, 3.0)]
)
def test_linear_regression_weights(weights: torch.Tensor | Sequence) -> None:
    assert LinearRegression(weights=weights).weights.equal(torch.tensor([2.0, 1.0, 3.0]))


@pytest.mark.parametrize("bias", [0.0, 1.0, -1.0])
def test_linear_regression_bias(bias: float) -> None:
    assert LinearRegression(weights=torch.tensor([2.0, 1.0, 3.0]), bias=bias).bias == bias


@pytest.mark.parametrize("noise_std", [0, 0.1, 1])
def test_linear_regression_noise_std(noise_std: float) -> None:
    assert LinearRegression.create_uniform_weights(noise_std=noise_std).noise_std == noise_std


@pytest.mark.parametrize("noise_std", [-1, -4.2])
def test_linear_regression_incorrect_noise_std(noise_std: float) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        LinearRegression.create_uniform_weights(noise_std=noise_std)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", [5, 8, 10])
def test_linear_regression_generate(batch_size: int, feature_size: int) -> None:
    data = LinearRegression.create_uniform_weights(feature_size=feature_size).generate(batch_size)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (batch_size, feature_size)
    assert data[ct.FEATURE].dtype == torch.float


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("bias", [0.0, 1.0])
def test_linear_regression_generate_same_random_seed(noise_std: float, bias: float) -> None:
    generator = LinearRegression.create_uniform_weights(noise_std=noise_std, bias=bias)
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("bias", [0.0, 1.0])
def test_linear_regression_generate_different_random_seeds(noise_std: float, bias: float) -> None:
    generator = LinearRegression.create_uniform_weights(noise_std=noise_std, bias=bias)
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("noise_std", [0.0, 1.0])
@pytest.mark.parametrize("weights", [torch.ones(3), torch.ones(5)])
@pytest.mark.parametrize("bias", [0.0, 1.0])
@pytest.mark.parametrize("rng", [None, get_torch_generator(1)])
def test_linear_regression_generate_mock(
    batch_size: int,
    noise_std: float,
    weights: int,
    bias: float,
    rng: torch.Generator | None,
) -> None:
    generator = LinearRegression(noise_std=noise_std, weights=weights, bias=bias)
    with patch("startorch.example.regression.make_linear_regression") as make_mock:
        generator.generate(batch_size=batch_size, rng=rng)
        make_mock.assert_called_once_with(
            num_examples=batch_size,
            noise_std=noise_std,
            weights=weights,
            bias=bias,
            generator=rng,
        )


############################################
#     Tests for make_linear_regression     #
############################################


@pytest.mark.parametrize("num_examples", [0, -1])
def test_make_linear_regression_incorrect_num_examples(num_examples: int) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for num_examples. Expected a value greater or equal to 1",
    ):
        make_linear_regression(weights=torch.ones(8), num_examples=num_examples)


@pytest.mark.parametrize("noise_std", [-1, -4.2])
def test_make_linear_regression_incorrect_noise_std(noise_std: float) -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect value for noise_std. Expected a value greater than 0",
    ):
        make_linear_regression(weights=torch.ones(8), noise_std=noise_std)


def test_make_linear_regression() -> None:
    data = make_linear_regression(num_examples=10, weights=torch.ones(8))
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (10, 8)
    assert data[ct.FEATURE].dtype == torch.float


def test_make_linear_regression_incorrect_weights() -> None:
    with pytest.raises(RuntimeError, match=r"shape '\[8, 1\]' is invalid for input of size"):
        make_linear_regression(num_examples=10, weights=torch.ones(8, 2))


@pytest.mark.parametrize("num_examples", SIZES)
def test_make_linear_regression_num_examples(num_examples: int) -> None:
    data = make_linear_regression(num_examples=num_examples, weights=torch.ones(10))
    assert len(data) == 2
    assert data[ct.TARGET].shape[0] == num_examples
    assert data[ct.FEATURE].shape[0] == num_examples


@pytest.mark.parametrize("feature_size", SIZES)
def test_make_linear_regression_feature_size(feature_size: int) -> None:
    data = make_linear_regression(num_examples=10, weights=torch.ones(feature_size))
    assert data[ct.FEATURE].shape == (10, feature_size)


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_make_linear_regression_same_random_seed(noise_std: float) -> None:
    weights = torch.rand(16)
    assert objects_are_equal(
        make_linear_regression(
            weights=weights, num_examples=64, noise_std=noise_std, generator=get_torch_generator(1)
        ),
        make_linear_regression(
            weights=weights, num_examples=64, noise_std=noise_std, generator=get_torch_generator(1)
        ),
    )


@pytest.mark.parametrize("noise_std", [0.0, 1.0])
def test_make_linear_regression_different_random_seeds(noise_std: float) -> None:
    weights = torch.rand(16)
    assert not objects_are_equal(
        make_linear_regression(
            weights=weights, num_examples=64, noise_std=noise_std, generator=get_torch_generator(1)
        ),
        make_linear_regression(
            weights=weights, num_examples=64, noise_std=noise_std, generator=get_torch_generator(2)
        ),
    )


#########################################
#     Tests for get_uniform_weights     #
#########################################


def test_get_uniform_weights_informative_feature_size_0() -> None:
    assert get_uniform_weights(feature_size=10, informative_feature_size=0).equal(torch.zeros(10))


def test_get_uniform_weights_informative_feature_size_10() -> None:
    assert not get_uniform_weights(feature_size=10, informative_feature_size=10).equal(
        torch.zeros(10)
    )


@pytest.mark.parametrize("informative_feature_size", [0, 1, 5, 10])
def test_get_uniform_weights_same_random_seed(informative_feature_size: int) -> None:
    assert objects_are_equal(
        get_uniform_weights(
            feature_size=8,
            informative_feature_size=informative_feature_size,
            generator=get_torch_generator(1),
        ),
        get_uniform_weights(
            feature_size=8,
            informative_feature_size=informative_feature_size,
            generator=get_torch_generator(1),
        ),
    )


@pytest.mark.parametrize("informative_feature_size", [1, 5, 10])
def test_get_uniform_weights_different_random_seeds(informative_feature_size: int) -> None:
    assert not objects_are_equal(
        get_uniform_weights(
            feature_size=8,
            informative_feature_size=informative_feature_size,
            generator=get_torch_generator(1),
        ),
        get_uniform_weights(
            feature_size=8,
            informative_feature_size=informative_feature_size,
            generator=get_torch_generator(2),
        ),
    )
