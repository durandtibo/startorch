from __future__ import annotations

import pytest
import torch

from startorch.tensor import Poisson, RandPoisson, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


#############################
#     Tests for Poisson     #
#############################


def test_poisson_str() -> None:
    assert str(Poisson(RandUniform())).startswith("PoissonTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_poisson_generate(size: tuple[int, ...]) -> None:
    tensor = Poisson(RandUniform(low=1.0, high=2.0)).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_poisson_generate_same_random_seed() -> None:
    generator = Poisson(RandUniform(low=1.0, high=2.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_poisson_generate_different_random_seeds() -> None:
    generator = Poisson(RandUniform(low=1.0, high=2.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#################################
#     Tests for RandPoisson     #
#################################


def test_rand_poisson_str() -> None:
    assert str(RandPoisson()).startswith("RandPoissonTensorGenerator(")


@pytest.mark.parametrize("rate", [1.0, 2.0])
def test_rand_poisson_rate(rate: float) -> None:
    assert RandPoisson(rate=rate)._rate == rate


def test_rand_poisson_rate_default() -> None:
    assert RandPoisson()._rate == 1.0


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_rand_poisson_rate_incorrect(rate: float) -> None:
    with pytest.raises(ValueError, match=r"rate has to be greater than 0"):
        RandPoisson(rate=rate)


@pytest.mark.parametrize("size", SIZES)
def test_rand_poisson_generate(size: tuple[int, ...]) -> None:
    tensor = RandPoisson().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0


def test_rand_poisson_generate_same_random_seed() -> None:
    generator = RandPoisson()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_poisson_generate_different_random_seeds() -> None:
    generator = RandPoisson()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
