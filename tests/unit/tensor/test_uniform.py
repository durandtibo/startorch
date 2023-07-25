from __future__ import annotations

from unittest.mock import Mock, patch

import torch
from pytest import mark, raises

from startorch.tensor import RandUniform, Uniform
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


#################################
#     Tests for RandUniform     #
#################################


def test_rand_uniform_str() -> None:
    assert str(RandUniform()).startswith("RandUniformTensorGenerator(")


@mark.parametrize("low", (1.0, 2.0))
def test_rand_uniform_low(low: float) -> None:
    assert RandUniform(low=low, high=10)._low == low


@mark.parametrize("high", (1.0, 10.0))
def test_rand_uniform_high(high: float) -> None:
    assert RandUniform(low=1, high=high)._high == high


def test_rand_uniform_incorrect_low_high() -> None:
    with raises(ValueError, match="high (.*) has to be greater or equal to low (.*)"):
        RandUniform(low=2, high=1)


@mark.parametrize("size", SIZES)
def test_rand_uniform_generate(size: tuple[int, ...]) -> None:
    tensor = RandUniform().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0
    assert tensor.max() < 1.0


def test_rand_uniform_generate_value_1() -> None:
    assert (
        RandUniform(low=1.0, high=1.0)
        .generate(size=(2, 4))
        .allclose(torch.ones(2, 4, dtype=torch.float))
    )


def test_rand_uniform_generate_mock() -> None:
    generator = RandUniform(0.0, 10.0)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.uniform.rand_uniform", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["size"] == (2, 4)
        assert mock.call_args.kwargs["low"] == 0.0
        assert mock.call_args.kwargs["high"] == 10.0


def test_rand_uniform_generate_same_random_seed() -> None:
    generator = RandUniform()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_uniform_generate_different_random_seeds() -> None:
    generator = RandUniform()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#############################
#     Tests for Uniform     #
#############################


def test_uniform_str() -> None:
    assert str(
        Uniform(
            low=RandUniform(low=-2.0, high=-1.0),
            high=RandUniform(low=1.0, high=2.0),
        )
    ).startswith("UniformTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_uniform_generate(size: tuple[int, ...]) -> None:
    tensor = Uniform(
        low=RandUniform(low=-2.0, high=-1.0),
        high=RandUniform(low=1.0, high=2.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= -2.0
    assert tensor.max() < 2.0


# def test_uniform_generate_value_1()->None:  # TODO
#     assert Uniform(low=Full(1.0), high=Full(1.0)).generate(size=(4, 12)).allclose(torch.ones(4, 12))


def test_uniform_generate_same_random_seed() -> None:
    generator = Uniform(low=RandUniform(low=-2.0, high=-1.0), high=RandUniform(low=1.0, high=2.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_uniform_generate_different_random_seeds() -> None:
    generator = Uniform(low=RandUniform(low=-2.0, high=-1.0), high=RandUniform(low=1.0, high=2.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
