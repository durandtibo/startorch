from __future__ import annotations

import torch
from pytest import mark

from startorch.tensor import Float, Full, Long, RandInt, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


###########################
#     Tests for Float     #
###########################


def test_float_str() -> None:
    assert str(Float(RandInt(low=0, high=10))).startswith("FloatTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_float_generate(size: tuple[int, ...]) -> None:
    assert Float(Full(value=4)).generate(size).equal(torch.full((size), 4.0, dtype=torch.float))


def test_float_generate_same_random_seed() -> None:
    generator = Float(RandInt(low=0, high=10))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_float_generate_different_random_seeds() -> None:
    generator = Float(RandInt(low=0, high=10))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


##########################
#     Tests for Long     #
##########################


def test_long_str() -> None:
    assert str(Long(RandUniform(low=0.0, high=50.0))).startswith("LongTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_long_generate(size: tuple[int, ...]) -> None:
    tensor = Long(RandUniform(low=0.0, high=50.0)).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.long
    assert tensor.min() >= 0
    assert tensor.max() < 50


def test_long_generate_same_random_seed() -> None:
    generator = Long(RandUniform(low=0.0, high=50.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_long_generate_different_random_seeds() -> None:
    generator = Long(RandUniform(low=0.0, high=50.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
