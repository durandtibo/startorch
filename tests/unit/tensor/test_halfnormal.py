from __future__ import annotations

from unittest.mock import Mock, patch

import torch
from pytest import mark, raises

from startorch.tensor import (
    HalfNormal,
    RandHalfNormal,
    RandTruncHalfNormal,
    RandUniform,
    TruncHalfNormal,
)
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


################################
#     Tests for HalfNormal     #
################################


def test_half_normal_str() -> None:
    assert str(HalfNormal(std=RandUniform(low=1.0, high=2.0))).startswith(
        "HalfNormalTensorGenerator("
    )


@mark.parametrize("size", SIZES)
def test_half_normal_generate(size: tuple[int, ...]) -> None:
    tensor = HalfNormal(std=RandUniform(low=1.0, high=2.0)).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_half_normal_generate_mock() -> None:
    generator = HalfNormal(std=RandUniform(low=1.0, high=2.0))
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.halfnormal.half_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_half_normal_generate_same_random_seed() -> None:
    generator = HalfNormal(std=RandUniform(low=1.0, high=2.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_half_normal_generate_different_random_seeds() -> None:
    generator = HalfNormal(std=RandUniform(low=1.0, high=2.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


####################################
#     Tests for RandHalfNormal     #
####################################


def test_rand_half_normal_str() -> None:
    assert str(RandHalfNormal()).startswith("RandHalfNormalTensorGenerator(")


@mark.parametrize("std", (1, 2))
def test_rand_half_normal_std(std: float) -> None:
    assert RandHalfNormal(std=std)._std == std


def test_rand_half_normal_std_default() -> None:
    assert RandHalfNormal()._std == 1.0


@mark.parametrize("std", (0, -1))
def test_rand_half_normal_incorrect_std(std: float) -> None:
    with raises(ValueError, match="std has to be greater than 0"):
        RandHalfNormal(std=std)


@mark.parametrize("size", SIZES)
def test_rand_half_normal_generate(size: tuple[int, ...]) -> None:
    tensor = RandHalfNormal().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0


@mark.parametrize("std", (0.1, 1.0))
def test_rand_half_normal_generate_mean_std(std: float) -> None:
    generator = RandHalfNormal(std=std)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.halfnormal.rand_half_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["std"] == std


def test_rand_half_normal_generate_same_random_seed() -> None:
    generator = RandHalfNormal()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_half_normal_generate_different_random_seeds() -> None:
    generator = RandHalfNormal()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#########################################
#     Tests for RandTruncHalfNormal     #
#########################################


def test_rand_trunc_normal_str() -> None:
    assert str(RandTruncHalfNormal()).startswith("RandTruncHalfNormalTensorGenerator(")


@mark.parametrize("std", (1.0, 2.0))
def test_rand_trunc_normal_std(std: float) -> None:
    assert RandTruncHalfNormal(std=std)._std == std


def test_rand_trunc_normal_std_default() -> None:
    assert RandTruncHalfNormal()._std == 1.0


@mark.parametrize("std", (0.0, -1.0))
def test_rand_trunc_normal_incorrect_std(std: float) -> None:
    with raises(ValueError, match="std has to be greater than 0"):
        RandTruncHalfNormal(std=std)


@mark.parametrize("max_value", (1.0, 2.0))
def test_rand_trunc_normal_max_value(max_value: float) -> None:
    assert RandTruncHalfNormal(max_value=max_value)._max_value == max_value


def test_rand_trunc_normal_max_value_default() -> None:
    assert RandTruncHalfNormal()._max_value == 3.0


@mark.parametrize("max_value", (0.0, -1.0))
def test_rand_trunc_normal_incorrect_max_value(max_value: float) -> None:
    with raises(ValueError, match="max_value has to be greater than 0"):
        RandTruncHalfNormal(max_value=max_value)


@mark.parametrize("size", SIZES)
def test_rand_trunc_normal_generate(size: tuple[int, ...]) -> None:
    tensor = RandTruncHalfNormal().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0
    assert tensor.max() < 3.0


@mark.parametrize("std", (0.1, 1.0))
def test_rand_trunc_normal_generate_std(std: float) -> None:
    generator = RandTruncHalfNormal(std=std)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.halfnormal.rand_trunc_half_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["std"] == std


@mark.parametrize("max_value", (2.0, 1.0))
def test_rand_trunc_normal_generate_min_max(max_value: float) -> None:
    generator = RandTruncHalfNormal(max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.halfnormal.rand_trunc_half_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["max_value"] == max_value


def test_rand_trunc_normal_generate_same_random_seed() -> None:
    generator = RandTruncHalfNormal()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_trunc_normal_generate_different_random_seeds() -> None:
    generator = RandTruncHalfNormal()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#####################################
#     Tests for TruncHalfNormal     #
#####################################


def test_trunc_half_normal_str() -> None:
    assert str(
        TruncHalfNormal(
            std=RandUniform(low=1.0, high=2.0),
            max_value=RandUniform(low=5.0, high=10.0),
        )
    ).startswith("TruncHalfNormalTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_trunc_half_normal_generate(size: tuple[int, ...]) -> None:
    tensor = TruncHalfNormal(
        std=RandUniform(low=1.0, high=2.0),
        max_value=RandUniform(low=5.0, high=10.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_trunc_half_normal_generate_mock() -> None:
    generator = TruncHalfNormal(
        std=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.halfnormal.trunc_half_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_trunc_half_normal_generate_same_random_seed() -> None:
    generator = TruncHalfNormal(
        std=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_trunc_half_normal_generate_different_random_seeds() -> None:
    generator = TruncHalfNormal(
        std=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
