from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from startorch.tensor import (
    AsinhUniform,
    Full,
    LogUniform,
    RandAsinhUniform,
    RandInt,
    RandLogUniform,
    RandUniform,
    Uniform,
)
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


##################################
#     Tests for AsinhUniform     #
##################################


def test_asinh_uniform_str() -> None:
    assert str(
        AsinhUniform(
            low=RandUniform(low=-1000.0, high=-1.0),
            high=RandUniform(low=1.0, high=1000.0),
        )
    ).startswith("AsinhUniformTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_asinh_uniform_generate(size: tuple[int, ...]) -> None:
    tensor = AsinhUniform(
        low=RandUniform(low=-1000.0, high=-1.0),
        high=RandUniform(low=1.0, high=1000.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= -1000.0
    assert tensor.max() < 1000.0


def test_asinh_uniform_generate_value_1() -> None:
    assert (
        AsinhUniform(low=Full(1.0), high=Full(1.0))
        .generate(size=(4, 12))
        .allclose(torch.ones(4, 12))
    )


def test_asinh_uniform_generate_same_random_seed() -> None:
    generator = AsinhUniform(
        low=RandUniform(low=-1000.0, high=-1.0), high=RandUniform(low=1.0, high=1000.0)
    )
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_asinh_uniform_generate_different_random_seeds() -> None:
    generator = AsinhUniform(
        low=RandUniform(low=-1000.0, high=-1.0), high=RandUniform(low=1.0, high=1000.0)
    )
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


################################
#     Tests for LogUniform     #
################################


def test_log_uniform_str() -> None:
    assert str(
        LogUniform(
            low=RandUniform(low=0.001, high=1.0),
            high=RandUniform(low=1.0, high=1000.0),
        )
    ).startswith("LogUniformTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_log_uniform_generate(size: tuple[int, ...]) -> None:
    tensor = LogUniform(
        low=RandUniform(low=0.001, high=1.0),
        high=RandUniform(low=1.0, high=1000.0),
    ).generate(size)
    assert tensor.data.shape == size
    assert tensor.data.dtype == torch.float
    assert tensor.min() >= 0.001
    assert tensor.max() < 1000.0


def test_log_uniform_generate_value_1() -> None:
    assert (
        LogUniform(low=Full(1.0), high=Full(1.0)).generate(size=(4, 12)).allclose(torch.ones(4, 12))
    )


def test_log_uniform_generate_same_random_seed() -> None:
    generator = LogUniform(
        low=RandUniform(low=0.001, high=1.0), high=RandUniform(low=1.0, high=1000.0)
    )
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_log_uniform_generate_different_random_seeds() -> None:
    generator = LogUniform(
        low=RandUniform(low=0.001, high=1.0), high=RandUniform(low=1.0, high=1000.0)
    )
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


######################################
#     Tests for RandAsinhUniform     #
######################################


def test_rand_asinh_uniform_str() -> None:
    assert str(RandAsinhUniform(low=-1000.0, high=1000.0)).startswith(
        "RandAsinhUniformTensorGenerator("
    )


@pytest.mark.parametrize("low", [1.0, 2.0])
def test_rand_asinh_uniform_low(low: float) -> None:
    assert RandAsinhUniform(low=low, high=10)._low == low


@pytest.mark.parametrize("high", [1.0, 10.0])
def test_rand_asinh_uniform_high(high: float) -> None:
    assert RandAsinhUniform(low=1, high=high)._high == high


def test_rand_asinh_uniform_incorrect_min_high() -> None:
    with pytest.raises(ValueError, match="high (.*) has to be greater or equal to low (.*)"):
        RandAsinhUniform(low=2, high=1)


@pytest.mark.parametrize("size", SIZES)
def test_rand_asinh_uniform_generate(size: tuple[int, ...]) -> None:
    tensor = RandAsinhUniform(low=-1000.0, high=1000.0).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= -1000
    assert tensor.max() < 1000.0


def test_rand_asinh_uniform_generate_mock() -> None:
    generator = RandAsinhUniform(low=-1.0, high=1.0)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.uniform.rand_asinh_uniform", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["size"] == (2, 4)
        assert mock.call_args.kwargs["low"] == -1.0
        assert mock.call_args.kwargs["high"] == 1.0


def test_rand_asinh_uniform_generate_same_random_seed() -> None:
    generator = RandAsinhUniform(low=-1000.0, high=1000.0)
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_asinh_uniform_generate_different_random_seeds() -> None:
    generator = RandAsinhUniform(low=-1000.0, high=1000.0)
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#############################
#     Tests for RandInt     #
#############################


def test_rand_int_str() -> None:
    assert str(RandInt(0, 10)).startswith("RandIntTensorGenerator(")


@pytest.mark.parametrize("low", [1, 2])
def test_rand_int_low(low: int) -> None:
    assert RandInt(low=low, high=10)._low == low


@pytest.mark.parametrize("high", [1, 10])
def test_rand_int_high(high: int) -> None:
    assert RandInt(low=0, high=high)._high == high


def test_rand_int_incorrect_low_high() -> None:
    with pytest.raises(ValueError, match="high (.*) has to be greater than low (.*)"):
        RandInt(low=1, high=1)


@pytest.mark.parametrize("size", SIZES)
def test_rand_int_generate(size: tuple[int, ...]) -> None:
    tensor = RandInt(0, 10).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.long
    assert tensor.min() >= 0
    assert tensor.max() < 10


def test_rand_int_generate_mock() -> None:
    generator = RandInt(0, 10)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.uniform.torch.randint", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["size"] == (2, 4)
        assert mock.call_args.kwargs["low"] == 0
        assert mock.call_args.kwargs["high"] == 10


def test_rand_int_generate_value_1() -> None:
    assert RandInt(low=1, high=2).generate(size=(2, 4)).allclose(torch.ones(2, 4, dtype=torch.long))


def test_rand_int_generate_same_random_seed() -> None:
    generator = RandInt(0, 10)
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_int_generate_different_random_seeds() -> None:
    generator = RandInt(0, 10)
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


####################################
#     Tests for RandLogUniform     #
####################################


def test_rand_log_uniform_str() -> None:
    assert str(RandLogUniform(low=0.001, high=1000.0)).startswith("RandLogUniformTensorGenerator(")


@pytest.mark.parametrize("low", [1.0, 2.0])
def test_rand_log_uniform_low(low: float) -> None:
    assert RandLogUniform(low=low, high=10.0)._low == low


@pytest.mark.parametrize("high", [1.0, 10.0])
def test_rand_log_uniform_high(high: float) -> None:
    assert RandLogUniform(low=0.1, high=high)._high == high


def test_rand_log_uniform_incorrect_min_high() -> None:
    with pytest.raises(ValueError, match="high (.*) has to be greater or equal to low"):
        RandLogUniform(low=2.0, high=1.0)


@pytest.mark.parametrize("size", SIZES)
def test_rand_log_uniform_generate(size: tuple[int, ...]) -> None:
    tensor = RandLogUniform(low=0.001, high=1000.0).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.001
    assert tensor.max() < 1000.0


def test_rand_log_uniform_value_1() -> None:
    assert RandLogUniform(low=1.0, high=1.0).generate(size=(4, 12)).allclose(torch.ones(4, 12))


def test_rand_log_uniform_generate_mock() -> None:
    generator = RandLogUniform(0.1, 10.0)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.uniform.rand_log_uniform", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["size"] == (2, 4)
        assert mock.call_args.kwargs["low"] == 0.1
        assert mock.call_args.kwargs["high"] == 10.0


def test_rand_log_uniform_generate_same_random_seed() -> None:
    generator = RandLogUniform(low=0.001, high=1000.0)
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_log_uniform_generate_different_random_seeds() -> None:
    generator = RandLogUniform(low=0.001, high=1000.0)
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#################################
#     Tests for RandUniform     #
#################################


def test_rand_uniform_str() -> None:
    assert str(RandUniform()).startswith("RandUniformTensorGenerator(")


@pytest.mark.parametrize("low", [1.0, 2.0])
def test_rand_uniform_low(low: float) -> None:
    assert RandUniform(low=low, high=10)._low == low


@pytest.mark.parametrize("high", [1.0, 10.0])
def test_rand_uniform_high(high: float) -> None:
    assert RandUniform(low=1, high=high)._high == high


def test_rand_uniform_incorrect_low_high() -> None:
    with pytest.raises(ValueError, match="high (.*) has to be greater or equal to low (.*)"):
        RandUniform(low=2, high=1)


@pytest.mark.parametrize("size", SIZES)
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


@pytest.mark.parametrize("size", SIZES)
def test_uniform_generate(size: tuple[int, ...]) -> None:
    tensor = Uniform(
        low=RandUniform(low=-2.0, high=-1.0),
        high=RandUniform(low=1.0, high=2.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= -2.0
    assert tensor.max() < 2.0


def test_uniform_generate_value_1() -> None:
    assert Uniform(low=Full(1.0), high=Full(1.0)).generate(size=(4, 12)).allclose(torch.ones(4, 12))


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
