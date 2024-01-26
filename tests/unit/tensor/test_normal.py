from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from startorch.tensor import (
    Normal,
    RandNormal,
    RandTruncNormal,
    RandUniform,
    TruncNormal,
)
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


############################
#     Tests for Normal     #
############################


def test_normal_str() -> None:
    assert str(
        Normal(mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0))
    ).startswith("NormalTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_normal_generate(size: tuple[int, ...]) -> None:
    tensor = Normal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_normal_generate_mock() -> None:
    generator = Normal(mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0))
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.normal.normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_normal_generate_same_random_seed() -> None:
    generator = Normal(mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_normal_generate_different_random_seeds() -> None:
    generator = Normal(mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


################################
#     Tests for RandNormal     #
################################


def test_rand_normal_str() -> None:
    assert str(RandNormal()).startswith("RandNormalTensorGenerator(")


@pytest.mark.parametrize("mean", [-1.0, 0.0, 1.0])
def test_rand_normal_mean(mean: float) -> None:
    assert RandNormal(mean=mean)._mean == mean


def test_rand_normal_mean_default() -> None:
    assert RandNormal()._mean == 0.0


@pytest.mark.parametrize("std", [1.0, 2.0])
def test_rand_normal_std(std: float) -> None:
    assert RandNormal(std=std)._std == std


def test_rand_normal_std_default() -> None:
    assert RandNormal()._std == 1.0


@pytest.mark.parametrize("std", [0.0, -1.0])
def test_rand_normal_incorrect_std(std: float) -> None:
    with pytest.raises(ValueError, match="std has to be greater than 0"):
        RandNormal(std=std)


@pytest.mark.parametrize("size", SIZES)
def test_rand_normal_generate(size: tuple[int, ...]) -> None:
    tensor = RandNormal().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


@pytest.mark.parametrize("mean", [0.0, 1.0])
@pytest.mark.parametrize("std", [0.1, 1.0])
def test_rand_normal_generate_mean_std(mean: float, std: float) -> None:
    generator = RandNormal(mean=mean, std=std)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.normal.rand_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["mean"] == mean
        assert mock.call_args.kwargs["std"] == std


def test_rand_normal_generate_same_random_seed() -> None:
    generator = RandNormal()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_normal_generate_different_random_seeds() -> None:
    generator = RandNormal()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#####################################
#     Tests for RandTruncNormal     #
#####################################


def test_rand_trunc_normal_str() -> None:
    assert str(RandTruncNormal()).startswith("RandTruncNormalTensorGenerator(")


@pytest.mark.parametrize("mean", [-1.0, 0.0, 1.0])
def test_rand_trunc_normal_mean(mean: float) -> None:
    assert RandTruncNormal(mean=mean)._mean == mean


def test_rand_trunc_normal_mean_default() -> None:
    assert RandTruncNormal()._mean == 0.0


@pytest.mark.parametrize("std", [1.0, 2.0])
def test_rand_trunc_normal_std(std: float) -> None:
    assert RandTruncNormal(std=std)._std == std


def test_rand_trunc_normal_std_default() -> None:
    assert RandTruncNormal()._std == 1.0


@pytest.mark.parametrize("std", [0.0, -1.0])
def test_rand_trunc_normal_incorrect_std(std: float) -> None:
    with pytest.raises(ValueError, match="std has to be greater than 0"):
        RandTruncNormal(std=std)


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
def test_rand_trunc_normal_min_value(min_value: float) -> None:
    assert RandTruncNormal(min_value=min_value)._min_value == min_value


def test_rand_trunc_normal_min_value_default() -> None:
    assert RandTruncNormal()._min_value == -3.0


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_rand_trunc_normal_max_value(max_value: float) -> None:
    assert RandTruncNormal(max_value=max_value)._max_value == max_value


def test_rand_trunc_normal_max_value_default() -> None:
    assert RandTruncNormal()._max_value == 3.0


def test_rand_trunc_normal_incorrect_min_max_value() -> None:
    with pytest.raises(ValueError, match="max_value (.*) has to be greater or equal to min_value"):
        RandTruncNormal(min_value=3, max_value=2)


@pytest.mark.parametrize("size", SIZES)
def test_rand_trunc_normal_generate(size: tuple[int, ...]) -> None:
    tensor = RandTruncNormal().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= -3.0
    assert tensor.max() <= 3.0


@pytest.mark.parametrize("mean", [0.0, 1.0])
@pytest.mark.parametrize("std", [0.1, 1.0])
def test_rand_trunc_normal_generate_mean_std(mean: float, std: float) -> None:
    generator = RandTruncNormal(mean=mean, std=std)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.normal.rand_trunc_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["mean"] == mean
        assert mock.call_args.kwargs["std"] == std


@pytest.mark.parametrize("min_value", [-2.0, -1.0])
@pytest.mark.parametrize("max_value", [2.0, 1.0])
def test_rand_trunc_normal_generate_min_max(min_value: float, max_value: float) -> None:
    generator = RandTruncNormal(min_value=min_value, max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.normal.rand_trunc_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["max_value"] == max_value
        assert mock.call_args.kwargs["min_value"] == min_value


def test_rand_trunc_normal_generate_same_random_seed() -> None:
    generator = RandTruncNormal()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_trunc_normal_generate_different_random_seeds() -> None:
    generator = RandTruncNormal()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#################################
#     Tests for TruncNormal     #
#################################


def test_trunc_normal_str() -> None:
    assert str(
        TruncNormal(
            mean=RandUniform(low=-1.0, high=1.0),
            std=RandUniform(low=1.0, high=2.0),
            min_value=RandUniform(low=-10.0, high=-5.0),
            max_value=RandUniform(low=5.0, high=10.0),
        )
    ).startswith("TruncNormalTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_trunc_normal_generate(size: tuple[int, ...]) -> None:
    tensor = TruncNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= -10.0
    assert tensor.max() <= 10.0


def test_trunc_normal_generate_mock() -> None:
    generator = TruncNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.normal.trunc_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_trunc_normal_generate_same_random_seed() -> None:
    generator = TruncNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_trunc_normal_generate_different_random_seeds() -> None:
    generator = TruncNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
