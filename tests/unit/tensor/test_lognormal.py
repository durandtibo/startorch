from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from startorch.tensor import (
    LogNormal,
    RandLogNormal,
    RandTruncLogNormal,
    RandUniform,
    TruncLogNormal,
)
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


###############################
#     Tests for LogNormal     #
###############################


def test_log_normal_str() -> None:
    assert str(
        LogNormal(mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0))
    ).startswith("LogNormalTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_log_normal_generate(size: tuple[int, ...]) -> None:
    tensor = LogNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_log_normal_generate_mock() -> None:
    generator = LogNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
    )
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.lognormal.log_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_log_normal_generate_same_random_seed() -> None:
    generator = LogNormal(mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_log_normal_generate_different_random_seeds() -> None:
    generator = LogNormal(mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


###################################
#     Tests for RandLogNormal     #
###################################


def test_rand_log_normal_str() -> None:
    assert str(RandLogNormal()).startswith("RandLogNormalTensorGenerator(")


@pytest.mark.parametrize("mean", [-1.0, 0.0, 1.0])
def test_rand_log_normal_mean(mean: float) -> None:
    assert RandLogNormal(mean=mean)._mean == mean


def test_rand_log_normal_mean_default() -> None:
    assert RandLogNormal()._mean == 0.0


@pytest.mark.parametrize("std", [1.0, 2.0])
def test_rand_log_normal_std(std: float) -> None:
    assert RandLogNormal(std=std)._std == std


def test_rand_log_normal_std_default() -> None:
    assert RandLogNormal()._std == 1.0


@pytest.mark.parametrize("std", [0.0, -1.0])
def test_rand_log_normal_incorrect_std(std: float) -> None:
    with pytest.raises(ValueError, match=r"std has to be greater than 0"):
        RandLogNormal(std=std)


@pytest.mark.parametrize("size", SIZES)
def test_rand_log_normal_generate_feature_size_tuple(size: tuple[int, ...]) -> None:
    tensor = RandLogNormal().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0


@pytest.mark.parametrize("mean", [1.0, 2.0])
@pytest.mark.parametrize("std", [1.0, 0.2])
def test_rand_log_normal_generate_mean_std(mean: float, std: float) -> None:
    generator = RandLogNormal(mean=mean, std=std)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.lognormal.rand_log_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["size"] == (2, 4)
        assert mock.call_args.kwargs["mean"] == mean
        assert mock.call_args.kwargs["std"] == std


def test_rand_log_normal_generate_same_random_seed() -> None:
    generator = RandLogNormal()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_log_normal_generate_different_random_seeds() -> None:
    generator = RandLogNormal()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


########################################
#     Tests for RandTruncLogNormal     #
########################################


def test_rand_trunc_log_normal_str() -> None:
    assert str(RandTruncLogNormal()).startswith("RandTruncLogNormalTensorGenerator(")


@pytest.mark.parametrize("mean", [-1.0, 0.0, 1.0])
def test_rand_trunc_log_normal_mean(mean: float) -> None:
    assert RandTruncLogNormal(mean=mean)._mean == mean


def test_rand_trunc_log_normal_mean_default() -> None:
    assert RandTruncLogNormal()._mean == 0.0


@pytest.mark.parametrize("std", [1.0, 2.0])
def test_rand_trunc_log_normal_std(std: float) -> None:
    assert RandTruncLogNormal(std=std)._std == std


def test_rand_trunc_log_normal_std_default() -> None:
    assert RandTruncLogNormal()._std == 1.0


@pytest.mark.parametrize("std", [0.0, -1.0])
def test_rand_trunc_log_normal_incorrect_std(std: float) -> None:
    with pytest.raises(ValueError, match=r"std has to be greater than 0"):
        RandTruncLogNormal(std=std)


@pytest.mark.parametrize("min_value", [0.0, 1.0])
def test_rand_trunc_log_normal_min_value(min_value: float) -> None:
    assert RandTruncLogNormal(min_value=min_value)._min_value == min_value


def test_rand_trunc_log_normal_min_value_default() -> None:
    assert RandTruncLogNormal()._min_value == 0.0


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_rand_trunc_log_normal_max_value(max_value: float) -> None:
    assert RandTruncLogNormal(max_value=max_value)._max_value == max_value


def test_rand_trunc_log_normal_max_value_default() -> None:
    assert RandTruncLogNormal()._max_value == 5.0


def test_rand_trunc_log_normal_incorrect_min_max_value() -> None:
    with pytest.raises(ValueError, match=r"max_value (.*) has to be greater or equal to min_value"):
        RandTruncLogNormal(min_value=3, max_value=2)


@pytest.mark.parametrize("size", SIZES)
def test_rand_trunc_log_normal_generate(size: tuple[int, ...]) -> None:
    tensor = RandTruncLogNormal().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0
    assert tensor.max() < 5.0


@pytest.mark.parametrize("mean", [0.0, 1.0])
@pytest.mark.parametrize("std", [0.1, 1.0])
def test_rand_trunc_log_normal_generate_mean_std(mean: float, std: float) -> None:
    generator = RandTruncLogNormal(mean=mean, std=std)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.lognormal.rand_trunc_log_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["size"] == (2, 4)
        assert mock.call_args.kwargs["mean"] == mean
        assert mock.call_args.kwargs["std"] == std


@pytest.mark.parametrize("min_value", [-2.0, -1.0])
@pytest.mark.parametrize("max_value", [2.0, 1.0])
def test_rand_trunc_log_normal_generate_min_max(min_value: float, max_value: float) -> None:
    generator = RandTruncLogNormal(min_value=min_value, max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.lognormal.rand_trunc_log_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["size"] == (2, 4)
        assert mock.call_args.kwargs["max_value"] == max_value
        assert mock.call_args.kwargs["min_value"] == min_value


def test_rand_trunc_log_normal_generate_same_random_seed() -> None:
    generator = RandTruncLogNormal()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_trunc_log_normal_generate_different_random_seeds() -> None:
    generator = RandTruncLogNormal()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


####################################
#     Tests for TruncLogNormal     #
####################################


def test_trunc_log_normal_str() -> None:
    assert str(
        TruncLogNormal(
            mean=RandUniform(low=-1.0, high=1.0),
            std=RandUniform(low=1.0, high=2.0),
            min_value=RandUniform(low=0.0, high=0.5),
            max_value=RandUniform(low=5.0, high=10.0),
        )
    ).startswith("TruncLogNormalTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_trunc_log_normal_generate(size: tuple[int, ...]) -> None:
    tensor = TruncLogNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=0.0, high=0.5),
        max_value=RandUniform(low=5.0, high=10.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_trunc_log_normal_generate_mock() -> None:
    generator = TruncLogNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=0.0, high=0.5),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.lognormal.trunc_log_normal", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_trunc_log_normal_generate_same_random_seed() -> None:
    generator = TruncLogNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=0.0, high=0.5),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_trunc_log_normal_generate_different_random_seeds() -> None:
    generator = TruncLogNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=0.0, high=0.5),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
