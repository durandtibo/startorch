from __future__ import annotations

from unittest.mock import Mock, patch

import torch
from pytest import mark, raises

from startorch.tensor import (
    Exponential,
    RandExponential,
    RandTruncExponential,
    RandUniform,
    TruncExponential,
)
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


#################################
#     Tests for Exponential     #
#################################


def test_exponential_str() -> None:
    assert str(Exponential(RandUniform(low=1.0, high=5.0))).startswith(
        "ExponentialTensorGenerator("
    )


@mark.parametrize("size", SIZES)
def test_exponential_generate(size: tuple[int, ...]) -> None:
    tensor = Exponential(rate=RandUniform(low=1.0, high=5.0)).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0


def test_exponential_generate_mock() -> None:
    generator = Exponential(rate=RandUniform(low=1.0, high=5.0))
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.exponential.exponential", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_exponential_generate_same_random_seed() -> None:
    generator = Exponential(rate=RandUniform(low=1.0, high=5.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_exponential_generate_different_random_seeds() -> None:
    generator = Exponential(rate=RandUniform(low=1.0, high=5.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#####################################
#     Tests for RandExponential     #
#####################################


def test_rand_exponential_str() -> None:
    assert str(RandExponential()).startswith("RandExponentialTensorGenerator(")


@mark.parametrize("rate", (1.0, 2.0))
def test_rand_exponential_rate(rate: float) -> None:
    assert RandExponential(rate=rate)._rate == rate


def test_rand_exponential_rate_default() -> None:
    assert RandExponential()._rate == 1.0


@mark.parametrize("rate", (0.0, -1.0))
def test_rand_exponential_incorrect_rate(rate: float) -> None:
    with raises(ValueError, match="rate has to be greater than 0"):
        RandExponential(rate=rate)


@mark.parametrize("size", SIZES)
def test_rand_exponential_generate(size: tuple[int, ...]) -> None:
    tensor = RandExponential().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0


@mark.parametrize("rate", (1, 2))
def test_rand_exponential_generate_rate(rate: float) -> None:
    generator = RandExponential(rate=rate)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.exponential.rand_exponential", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["rate"] == rate


def test_rand_exponential_generate_same_random_seed() -> None:
    generator = RandExponential()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_exponential_generate_different_random_seeds() -> None:
    generator = RandExponential()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


##########################################
#     Tests for RandTruncExponential     #
##########################################


def test_rand_trunc_exponential_str() -> None:
    assert str(RandTruncExponential()).startswith("RandTruncExponentialTensorGenerator(")


@mark.parametrize("rate", (1.0, 2.0))
def test_rand_trunc_exponential_rate(rate: float) -> None:
    assert RandTruncExponential(rate=rate)._rate == rate


def test_rand_trunc_exponential_rate_default() -> None:
    assert RandTruncExponential()._rate == 1.0


@mark.parametrize("rate", (0.0, -1.0))
def test_rand_trunc_exponential_incorrect_rate(rate: float) -> None:
    with raises(ValueError, match="rate has to be greater than 0"):
        RandTruncExponential(rate=rate)


@mark.parametrize("max_value", (1.0, 2.0))
def test_rand_trunc_exponential_max_value(max_value: float) -> None:
    assert RandTruncExponential(max_value=max_value)._max_value == max_value


def test_rand_trunc_exponential_max_value_default() -> None:
    assert RandTruncExponential()._max_value == 5.0


@mark.parametrize("max_value", (0.0, -1.0))
def test_rand_trunc_exponential_incorrect_max_value(max_value: float) -> None:
    with raises(ValueError, match="max_value has to be greater than 0"):
        RandTruncExponential(max_value=max_value)


@mark.parametrize("size", SIZES)
def test_rand_trunc_exponential_generate(size: tuple[int, ...]) -> None:
    tensor = RandTruncExponential().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0
    assert tensor.max() <= 5.0


@mark.parametrize("rate", (1, 2))
def test_rand_trunc_exponential_generate_rate(rate: float) -> None:
    generator = RandTruncExponential(rate=rate)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.exponential.rand_trunc_exponential", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["rate"] == rate


@mark.parametrize("max_value", (1, 2))
def test_rand_trunc_exponential_generate_max_value(max_value: float) -> None:
    generator = RandTruncExponential(max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.exponential.rand_trunc_exponential", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["max_value"] == max_value


def test_rand_trunc_exponential_generate_same_random_seed() -> None:
    generator = RandTruncExponential()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_trunc_exponential_generate_different_random_seeds() -> None:
    generator = RandTruncExponential()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


######################################
#     Tests for TruncExponential     #
######################################


def test_trunc_exponential_str() -> None:
    assert str(
        TruncExponential(
            rate=RandUniform(low=1.0, high=2.0),
            max_value=RandUniform(low=5.0, high=10.0),
        )
    ).startswith("TruncExponentialTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_trunc_exponential_generate(size: tuple[int, ...]) -> None:
    tensor = TruncExponential(
        rate=RandUniform(low=1.0, high=2.0),
        max_value=RandUniform(low=5.0, high=10.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0
    assert tensor.max() <= 10.0


def test_trunc_exponential_generate_mock() -> None:
    generator = TruncExponential(
        rate=RandUniform(low=1.0, high=5.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.exponential.trunc_exponential", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_trunc_exponential_generate_same_random_seed() -> None:
    generator = TruncExponential(
        rate=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_trunc_exponential_generate_different_random_seeds() -> None:
    generator = TruncExponential(
        rate=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
