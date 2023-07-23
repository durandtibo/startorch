from __future__ import annotations

import math

import torch
from pytest import mark, raises

from startorch.random import (
    exponential,
    half_cauchy,
    half_normal,
    log_normal,
    rand_exponential,
    rand_half_cauchy,
    rand_half_normal,
    rand_log_normal,
)
from startorch.utils.seed import get_torch_generator

TOLERANCE = 0.05

######################################
#     Tests for rand_exponential     #
######################################


def test_rand_exponential_1d() -> None:
    values = rand_exponential((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(2.0).log(), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.min() >= 0.0


def test_rand_exponential_2d() -> None:
    values = rand_exponential((1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(2.0).log(), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.min() >= 0.0


@mark.parametrize("rate", (0.5, 2.0))
def test_rand_exponential_rate(rate: float) -> None:
    values = rand_exponential((100000,), rate=rate, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(1.0).div(rate), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(2.0).log().div(rate), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0).div(rate**2).sqrt(), atol=TOLERANCE)
    assert values.min() >= 0.0


def test_rand_exponential_incorrect_rate() -> None:
    with raises(RuntimeError, match="rate has to be greater than 0"):
        rand_exponential((1000,), rate=0, generator=get_torch_generator(1))


def test_rand_exponential_same_random_seed() -> None:
    assert rand_exponential((1000,), generator=get_torch_generator(1)).equal(
        rand_exponential((1000,), generator=get_torch_generator(1))
    )


def test_rand_exponential_different_random_seeds() -> None:
    assert not rand_exponential((1000,), generator=get_torch_generator(1)).equal(
        rand_exponential((1000,), generator=get_torch_generator(2))
    )


#################################
#     Tests for exponential     #
#################################


def test_exponential_1d() -> None:
    values = exponential(torch.ones(100000), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(2.0).log(), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.min() >= 0.0


def test_exponential_2d() -> None:
    values = exponential(torch.ones(1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(2.0).log(), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.min() >= 0.0


@mark.parametrize("rate", (0.5, 2.0))
def test_exponential_rate(rate: float) -> None:
    values = exponential(torch.full((100000,), rate), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(1.0).div(rate), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(2.0).log().div(rate), atol=TOLERANCE)
    assert values.std().allclose(torch.tensor(1.0).div(rate**2).sqrt(), atol=TOLERANCE)
    assert values.min() >= 0.0


def test_exponential_incorrect_rate() -> None:
    with raises(RuntimeError, match="rate values have to be greater than 0"):
        exponential(torch.zeros(1000), generator=get_torch_generator(1))


def test_exponential_same_random_seed() -> None:
    assert exponential(torch.ones(1000), generator=get_torch_generator(1)).equal(
        exponential(torch.ones(1000), generator=get_torch_generator(1))
    )


def test_exponential_different_random_seeds() -> None:
    assert not exponential(torch.ones(1000), generator=get_torch_generator(1)).equal(
        exponential(torch.ones(1000), generator=get_torch_generator(2))
    )


######################################
#     Tests for rand_half_cauchy     #
######################################


def test_rand_half_cauchy_1d() -> None:
    values = rand_half_cauchy((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0


def test_rand_half_cauchy_2d() -> None:
    values = rand_half_cauchy((1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.min() >= 0.0


def test_rand_half_cauchy_scale() -> None:
    assert not rand_half_cauchy(
        (100000,),
        scale=1.0,
        generator=get_torch_generator(1),
    ).equal(
        rand_half_cauchy(
            (100000,),
            scale=2.0,
            generator=get_torch_generator(1),
        )
    )


@mark.parametrize("scale", (0.0, -1.0))
def test_rand_half_cauchy_scale_incorrect(scale: float) -> None:
    with raises(RuntimeError, match="scale has to be greater than 0"):
        rand_half_cauchy((1000,), scale=scale, generator=get_torch_generator(1))


def test_rand_half_cauchy_same_random_seed() -> None:
    assert rand_half_cauchy((1000,), generator=get_torch_generator(1)).equal(
        rand_half_cauchy((1000,), generator=get_torch_generator(1))
    )


def test_rand_half_cauchy_different_random_seeds() -> None:
    assert not rand_half_cauchy((1000,), generator=get_torch_generator(1)).equal(
        rand_half_cauchy((1000,), generator=get_torch_generator(2))
    )


#################################
#     Tests for half_cauchy     #
#################################


def test_half_cauchy_1d() -> None:
    values = half_cauchy(torch.ones(100000), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0


def test_half_cauchy_2d() -> None:
    values = half_cauchy(torch.ones(1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.min() >= 0.0


def test_half_cauchy_scale() -> None:
    assert not half_cauchy(
        torch.ones(100000),
        generator=get_torch_generator(1),
    ).equal(
        half_cauchy(
            torch.full((100000,), 2.0),
            generator=get_torch_generator(1),
        )
    )


@mark.parametrize("scale", (0.0, -1.0))
def test_half_cauchy_scale_incorrect(scale: float) -> None:
    with raises(RuntimeError, match="scale has to be greater than 0"):
        half_cauchy(torch.full((1000,), scale), generator=get_torch_generator(1))


def test_half_cauchy_same_random_seed() -> None:
    assert half_cauchy(torch.ones(1000), generator=get_torch_generator(1)).equal(
        half_cauchy(torch.ones(1000), generator=get_torch_generator(1))
    )


def test_half_cauchy_different_random_seeds() -> None:
    assert not half_cauchy(torch.ones(1000), generator=get_torch_generator(1)).equal(
        half_cauchy(torch.ones(1000), generator=get_torch_generator(2))
    )


######################################
#     Tests for rand_half_normal     #
######################################


def test_rand_half_normal_1d() -> None:
    values = rand_half_normal((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(math.sqrt(2.0 / math.pi)), atol=TOLERANCE)
    assert values.min() >= 0.0


def test_rand_half_normal_2d() -> None:
    values = rand_half_normal((1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(math.sqrt(2.0 / math.pi)), atol=TOLERANCE)
    assert values.min() >= 0.0


@mark.parametrize("std", (0.5, 1.0))
def test_rand_half_normal_rate(std: float) -> None:
    values = rand_half_normal((100000,), std=std, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(math.sqrt(2.0 / math.pi)).mul(std), atol=TOLERANCE)
    assert values.min() >= 0.0


@mark.parametrize("std", (0.0, -1.0))
def test_rand_half_normal_std_incorrect(std: float) -> None:
    with raises(RuntimeError, match="std has to be greater than 0"):
        rand_half_normal((1000,), std=std, generator=get_torch_generator(1))


def test_rand_half_normal_same_random_seed() -> None:
    assert rand_half_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_half_normal((1000,), generator=get_torch_generator(1))
    )


def test_rand_half_normal_different_random_seeds() -> None:
    assert not rand_half_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_half_normal((1000,), generator=get_torch_generator(2))
    )


#################################
#     Tests for half_normal     #
#################################


def test_half_normal_1d() -> None:
    values = half_normal(torch.ones(100000), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(math.sqrt(2.0 / math.pi)), atol=TOLERANCE)
    assert values.min() >= 0.0


def test_half_normal_2d() -> None:
    values = half_normal(torch.ones(1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(math.sqrt(2.0 / math.pi)), atol=TOLERANCE)
    assert values.min() >= 0.0


@mark.parametrize("std", (0.5, 1.0))
def test_half_normal_std(std: float) -> None:
    values = half_normal(torch.full((100000,), std), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(math.sqrt(2.0 / math.pi)).mul(std), atol=TOLERANCE)
    assert values.min() >= 0.0


@mark.parametrize("std", (0.0, -1.0))
def test_half_normal_std_incorrect(std: float) -> None:
    with raises(RuntimeError, match="std has to be greater than 0"):
        half_normal(torch.full((1000,), std), generator=get_torch_generator(1))


def test_half_normal_same_random_seed() -> None:
    assert half_normal(torch.ones(1000), generator=get_torch_generator(1)).equal(
        half_normal(torch.ones(1000), generator=get_torch_generator(1))
    )


def test_half_normal_different_random_seeds() -> None:
    assert not half_normal(torch.ones(1000), generator=get_torch_generator(1)).equal(
        half_normal(torch.ones(1000), generator=get_torch_generator(2))
    )


#####################################
#     Tests for rand_log_normal     #
#####################################


def test_rand_log_normal_1d() -> None:
    values = rand_log_normal((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.5).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0), rtol=TOLERANCE)
    assert values.std().allclose(
        torch.tensor(1.0).exp().sub(1.0).mul(torch.tensor(1.0).exp()).sqrt(), rtol=TOLERANCE
    )
    assert values.min() >= 0.0


def test_rand_log_normal_2d() -> None:
    values = rand_log_normal((1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.5).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0), rtol=TOLERANCE)
    assert values.std().allclose(
        torch.tensor(1.0).exp().sub(1.0).mul(torch.tensor(1.0).exp()).sqrt(), rtol=TOLERANCE
    )
    assert values.min() >= 0.0


@mark.parametrize("mean", (-1.0, 0.0, 1.0))
def test_rand_log_normal_mean(mean: float) -> None:
    values = rand_log_normal((100000,), mean=mean, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(mean + 0.5).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(mean).exp(), rtol=TOLERANCE)
    assert values.std().allclose(
        torch.tensor(1.0).exp().sub(1.0).mul(torch.tensor(2.0 * mean + 1.0).exp()).sqrt(),
        rtol=TOLERANCE,
    )
    assert values.min() >= 0.0


@mark.parametrize("std", (0.25, 0.5, 1.0))
def test_rand_log_normal_std(std: float) -> None:
    values = rand_log_normal((100000,), std=std, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.5 * std**2).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0), rtol=TOLERANCE)
    assert values.std().allclose(
        torch.tensor(std**2).exp().sub(1.0).mul(torch.tensor(std**2).exp()).sqrt(),
        rtol=TOLERANCE,
    )
    assert values.min() >= 0.0


@mark.parametrize("std", (0.0, -1.0))
def test_rand_log_normal_std_incorrect(std: float) -> None:
    with raises(RuntimeError, match="std has to be greater than 0"):
        rand_log_normal((1000,), std=std, generator=get_torch_generator(1))


def test_rand_log_normal_same_random_seed() -> None:
    assert rand_log_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_log_normal((1000,), generator=get_torch_generator(1))
    )


def test_rand_log_normal_different_random_seeds() -> None:
    assert not rand_log_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_log_normal((1000,), generator=get_torch_generator(2))
    )


################################
#     Tests for log_normal     #
################################


def test_log_normal_1d() -> None:
    values = log_normal(
        torch.ones(100000), torch.full((100000,), 2.0), generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(3.0).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0).exp(), rtol=TOLERANCE)
    assert values.min() >= 0.0


def test_log_normal_2d() -> None:
    values = log_normal(
        torch.ones(1000, 100), torch.full((1000, 100), 2.0), generator=get_torch_generator(1)
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(3.0).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0).exp(), rtol=TOLERANCE)
    assert values.min() >= 0.0


@mark.parametrize("mean", (1.0, 0.1))
@mark.parametrize("std", (1.0, 0.1))
def test_log_normal_mean_std(mean: float, std: float) -> None:
    values = log_normal(
        torch.full((100000,), mean), torch.full((100000,), std), generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(mean + 0.5 * std**2).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(mean).exp(), rtol=TOLERANCE)
    assert values.min() >= 0.0


@mark.parametrize("std", (0.0, -1.0))
def test_log_normal_std_incorrect(std: float) -> None:
    with raises(RuntimeError, match="std has to be greater than 0"):
        log_normal(torch.ones(1000), torch.full((1000,), std), generator=get_torch_generator(1))


def test_log_normal_same_random_seed() -> None:
    assert log_normal(
        torch.ones(1000), torch.full((1000,), 2.0), generator=get_torch_generator(1)
    ).equal(
        log_normal(torch.ones(1000), torch.full((1000,), 2.0), generator=get_torch_generator(1))
    )


def test_log_normal_different_random_seeds() -> None:
    assert not log_normal(
        torch.ones(1000), torch.full((1000,), 2.0), generator=get_torch_generator(1)
    ).equal(
        log_normal(torch.ones(1000), torch.full((1000,), 2.0), generator=get_torch_generator(2))
    )
