from __future__ import annotations

import torch
from pytest import mark, raises

from startorch.random import rand_poisson
from startorch.utils.seed import get_torch_generator

TOLERANCE = 0.05

##################################
#     Tests for rand_poisson     #
##################################


def test_rand_poisson_rate_1d() -> None:
    tensor = rand_poisson((100000,), generator=get_torch_generator(1))
    assert tensor.shape == (100000,)
    assert tensor.dtype == torch.float
    assert torch.isclose(tensor.mean(), torch.tensor(1.0), rtol=TOLERANCE)
    assert torch.isclose(tensor.std(), torch.tensor(1.0).sqrt(), rtol=TOLERANCE)
    assert tensor.min() >= 0.0


def test_rand_poisson_rate_2d() -> None:
    tensor = rand_poisson((1000, 100), generator=get_torch_generator(1))
    assert tensor.shape == (1000, 100)
    assert tensor.dtype == torch.float
    assert torch.isclose(tensor.mean(), torch.tensor(1.0), rtol=TOLERANCE)
    assert torch.isclose(tensor.std(), torch.tensor(1.0).sqrt(), rtol=TOLERANCE)
    assert tensor.min() >= 0.0


@mark.parametrize("rate", (1.0, 0.1))
def test_rand_poisson_rate(rate: float) -> None:
    tensor = rand_poisson((100000,), rate=rate, generator=get_torch_generator(1))
    assert tensor.shape == (100000,)
    assert tensor.dtype == torch.float
    assert torch.isclose(tensor.mean(), torch.tensor(rate), rtol=TOLERANCE)
    assert torch.isclose(tensor.std(), torch.tensor(rate).sqrt(), rtol=TOLERANCE)
    assert tensor.min() >= 0.0


@mark.parametrize("rate", (0.0, -1.0))
def test_rand_poisson_incorrect_rate(rate: float) -> None:
    with raises(ValueError, match="rate has to be greater than 0"):
        rand_poisson((1000,), rate=rate, generator=get_torch_generator(1))


def test_rand_poisson_same_seed() -> None:
    assert rand_poisson((1000,), generator=get_torch_generator(1)).equal(
        rand_poisson((1000,), generator=get_torch_generator(1))
    )


def test_rand_poisson_different_seeds() -> None:
    assert not rand_poisson((1000,), generator=get_torch_generator(1)).equal(
        rand_poisson((1000,), generator=get_torch_generator(2))
    )
