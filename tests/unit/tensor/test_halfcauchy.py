from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from startorch.tensor import (
    HalfCauchy,
    RandHalfCauchy,
    RandTruncHalfCauchy,
    RandUniform,
    TruncHalfCauchy,
)
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


################################
#     Tests for HalfCauchy     #
################################


def test_half_cauchy_str() -> None:
    assert str(HalfCauchy(scale=RandUniform(low=1.0, high=2.0))).startswith(
        "HalfCauchyTensorGenerator("
    )


@pytest.mark.parametrize("size", SIZES)
def test_half_cauchy_generate(size: tuple[int, ...]) -> None:
    tensor = HalfCauchy(scale=RandUniform(low=1.0, high=2.0)).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_half_cauchy_generate_mock() -> None:
    generator = HalfCauchy(scale=RandUniform(low=1.0, high=2.0))
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.halfcauchy.half_cauchy", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_half_cauchy_generate_same_random_seed() -> None:
    generator = HalfCauchy(scale=RandUniform(low=1.0, high=2.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_half_cauchy_generate_different_random_seeds() -> None:
    generator = HalfCauchy(scale=RandUniform(low=1.0, high=2.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


####################################
#     Tests for RandHalfCauchy     #
####################################


def test_rand_half_cauchy_str() -> None:
    assert str(RandHalfCauchy()).startswith("RandHalfCauchyTensorGenerator(")


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rand_half_cauchy_scale(scale: float) -> None:
    assert RandHalfCauchy(scale=scale)._scale == scale


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_rand_half_cauchy_incorrect_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="scale has to be greater than 0"):
        RandHalfCauchy(scale=scale)


@pytest.mark.parametrize("size", SIZES)
def test_rand_half_cauchy_generate(size: tuple[int, ...]) -> None:
    tensor = RandHalfCauchy().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rand_half_cauchy_generate_scale(scale: float) -> None:
    generator = RandHalfCauchy(scale=scale)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.halfcauchy.rand_half_cauchy", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["scale"] == scale


def test_rand_half_cauchy_generate_same_random_seed() -> None:
    generator = RandHalfCauchy()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_half_cauchy_generate_different_random_seeds() -> None:
    generator = RandHalfCauchy()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#########################################
#     Tests for RandTruncHalfCauchy     #
#########################################


def test_rand_trunc_half_cauchy_str() -> None:
    assert str(RandTruncHalfCauchy()).startswith("RandTruncHalfCauchyTensorGenerator(")


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rand_trunc_half_cauchy_scale(scale: float) -> None:
    assert RandTruncHalfCauchy(scale=scale)._scale == scale


def test_rand_trunc_half_cauchy_scale_default() -> None:
    assert RandTruncHalfCauchy()._scale == 1.0


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_rand_trunc_half_cauchy_incorrect_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="scale has to be greater than 0"):
        RandTruncHalfCauchy(scale=scale)


@pytest.mark.parametrize("max_value", [1, 2])
def test_rand_trunc_half_cauchy_max_value(max_value: float) -> None:
    assert RandTruncHalfCauchy(max_value=max_value)._max_value == max_value


def test_rand_trunc_half_cauchy_max_value_default() -> None:
    assert RandTruncHalfCauchy()._max_value == 4.0


def test_rand_trunc_half_cauchy_incorrect_max_value() -> None:
    with pytest.raises(ValueError, match="max_value has to be greater than 0"):
        RandTruncHalfCauchy(max_value=0.0)


@pytest.mark.parametrize("size", SIZES)
def test_rand_trunc_half_cauchy_generate(size: tuple[int, ...]) -> None:
    tensor = RandTruncHalfCauchy().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0
    assert tensor.max() <= 4


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rand_trunc_half_cauchy_generate_scale(scale: float) -> None:
    generator = RandTruncHalfCauchy(scale=scale)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.halfcauchy.rand_trunc_half_cauchy", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["scale"] == scale


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_rand_trunc_half_cauchy_generate_max_value(max_value: float) -> None:
    generator = RandTruncHalfCauchy(max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.halfcauchy.rand_trunc_half_cauchy", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["max_value"] == max_value


def test_rand_trunc_half_cauchy_generate_same_random_seed() -> None:
    generator = RandTruncHalfCauchy()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_trunc_half_cauchy_generate_different_random_seeds() -> None:
    generator = RandTruncHalfCauchy()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#####################################
#     Tests for TruncHalfCauchy     #
#####################################


def test_trunc_half_cauchy_str() -> None:
    assert str(
        TruncHalfCauchy(
            scale=RandUniform(low=1.0, high=2.0),
            max_value=RandUniform(low=5.0, high=10.0),
        )
    ).startswith("TruncHalfCauchyTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_trunc_half_cauchy_generate(size: tuple[int, ...]) -> None:
    tensor = TruncHalfCauchy(
        scale=RandUniform(low=1.0, high=2.0),
        max_value=RandUniform(low=5.0, high=10.0),
    ).generate(size)
    assert tensor.data.shape == size
    assert tensor.data.dtype == torch.float


def test_trunc_half_cauchy_generate_mock() -> None:
    generator = TruncHalfCauchy(
        scale=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.halfcauchy.trunc_half_cauchy", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_trunc_half_cauchy_generate_same_random_seed() -> None:
    generator = TruncHalfCauchy(
        scale=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_trunc_half_cauchy_generate_different_random_seeds() -> None:
    generator = TruncHalfCauchy(
        scale=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
