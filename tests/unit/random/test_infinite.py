from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from startorch.random import cauchy, normal, rand_cauchy, rand_normal
from startorch.utils.seed import get_torch_generator

TOLERANCE = 0.05


#################################
#     Tests for rand_cauchy     #
#################################


def test_rand_cauchy_1d() -> None:
    values = rand_cauchy((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)


def test_rand_cauchy_2d() -> None:
    values = rand_cauchy((1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)


@pytest.mark.parametrize("loc", [-1.0, 0.0, 1.0])
def test_rand_cauchy_loc(loc: float) -> None:
    values = rand_cauchy((100000,), loc=loc, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(loc), atol=TOLERANCE)


def test_rand_cauchy_scale() -> None:
    assert not torch.allclose(
        rand_cauchy((100000,), scale=1.0, generator=get_torch_generator(1)).std(),
        rand_cauchy((100000,), scale=0.1, generator=get_torch_generator(1)).std(),
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_rand_cauchy_scale_incorrect(scale: float) -> None:
    with pytest.raises(ValueError, match="scale has to be greater than 0"):
        rand_cauchy((1000,), scale=scale, generator=get_torch_generator(1))


def test_rand_cauchy_same_random_seed() -> None:
    assert rand_cauchy((1000,), generator=get_torch_generator(1)).equal(
        rand_cauchy((1000,), generator=get_torch_generator(1))
    )


def test_rand_cauchy_different_random_seeds() -> None:
    assert not rand_cauchy((1000,), generator=get_torch_generator(1)).equal(
        rand_cauchy((1000,), generator=get_torch_generator(2))
    )


############################
#     Tests for cauchy     #
############################


def test_cauchy_1d() -> None:
    values = cauchy(torch.zeros(100000), torch.ones(100000), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)


def test_cauchy_2d() -> None:
    values = cauchy(torch.zeros(1000, 100), torch.ones(1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)


@pytest.mark.parametrize("loc", [-1.0, 0.0, 1.0])
def test_cauchy_loc(loc: float) -> None:
    values = cauchy(
        torch.full((100000,), loc), torch.ones(100000), generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(loc), atol=TOLERANCE)


def test_cauchy_scale() -> None:
    assert not torch.allclose(
        cauchy(
            torch.zeros(100000), torch.full((100000,), 1.0), generator=get_torch_generator(1)
        ).std(),
        cauchy(
            torch.zeros(100000), torch.full((100000,), 0.1), generator=get_torch_generator(1)
        ).std(),
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_cauchy_scale_incorrect(scale: float) -> None:
    with pytest.raises(ValueError, match="scale has to be greater than 0"):
        cauchy(torch.zeros(100000), torch.full((100000,), scale), generator=get_torch_generator(1))


def test_cauchy_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="The shapes of loc and scale do not match"):
        cauchy(torch.zeros(5), torch.ones(6), generator=get_torch_generator(1))


def test_cauchy_same_random_seed() -> None:
    assert cauchy(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(1)).equal(
        cauchy(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(1))
    )


def test_cauchy_different_random_seeds() -> None:
    assert not cauchy(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(1)).equal(
        cauchy(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(2))
    )


#################################
#     Tests for rand_normal     #
#################################


def test_rand_normal_1d() -> None:
    values = rand_normal((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0), atol=TOLERANCE)


def test_rand_normal_2d() -> None:
    values = rand_normal((10000, 10), generator=get_torch_generator(1))
    assert values.shape == (10000, 10)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0), atol=TOLERANCE)


@pytest.mark.parametrize("mean", [-1.0, 0.0, 1.0])
def test_rand_normal_mean(mean: float) -> None:
    values = rand_normal((100000,), mean=mean, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(mean), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0), atol=TOLERANCE)


@pytest.mark.parametrize("std", [0.1, 0.5, 1.0])
def test_rand_normal_std(std: float) -> None:
    values = rand_normal((100000,), std=std, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(std), atol=TOLERANCE)


@pytest.mark.parametrize("std", [0.0, -1.0])
def test_rand_normal_std_incorrect(std: float) -> None:
    with pytest.raises(ValueError, match="std has to be greater than 0"):
        rand_normal((1000,), std=std, generator=get_torch_generator(1))


def test_rand_normal_mock() -> None:
    with patch(
        "startorch.random.infinite.torch.randn",
        Mock(return_value=torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])),
    ):
        assert rand_normal((5,), mean=1.0, std=2.0, generator=get_torch_generator(1)).equal(
            torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
        )


def test_rand_normal_same_random_seed() -> None:
    assert rand_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_normal((1000,), generator=get_torch_generator(1))
    )


def test_rand_normal_different_random_seeds() -> None:
    assert not rand_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_normal((1000,), generator=get_torch_generator(2))
    )


############################
#     Tests for normal     #
############################


def test_normal_1d() -> None:
    values = normal(torch.zeros(100000), torch.ones(100000), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0), atol=TOLERANCE)


def test_normal_2d() -> None:
    values = normal(torch.zeros(1000, 100), torch.ones(1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0), atol=TOLERANCE)


@pytest.mark.parametrize("mean", [-1.0, 0.0, 1.0])
def test_normal_mean(mean: float) -> None:
    values = normal(
        torch.full((100000,), mean), torch.ones(100000), generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(mean), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0), atol=TOLERANCE)


@pytest.mark.parametrize("std", [0.1, 0.5, 1.0])
def test_normal_std(std: float) -> None:
    values = normal(
        torch.zeros(100000), torch.full((100000,), std), generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(std), atol=TOLERANCE)


def test_normal_mock() -> None:
    with patch(
        "startorch.random.infinite.torch.randn",
        Mock(return_value=torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])),
    ):
        assert normal(torch.ones(5), torch.full((5,), 2.0), generator=get_torch_generator(1)).equal(
            torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
        )


@pytest.mark.parametrize("std", [0.0, -1.0])
def test_normal_std_incorrect(std: float) -> None:
    with pytest.raises(ValueError, match="std has to be greater than 0"):
        normal(torch.zeros(1000), torch.full((1000,), std))


def test_normal_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="The shapes of mean and std do not match"):
        normal(torch.zeros(5), torch.ones(6), generator=get_torch_generator(1))


def test_normal_same_random_seed() -> None:
    assert normal(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(1)).equal(
        normal(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(1))
    )


def test_normal_different_random_seeds() -> None:
    assert not normal(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(1)).equal(
        normal(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(2))
    )
