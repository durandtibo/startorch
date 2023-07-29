from __future__ import annotations

from unittest.mock import Mock, patch

import torch
from pytest import mark, raises
from redcat import BatchedTensorSeq

from startorch.sequence import (
    LogNormal,
    RandLogNormal,
    RandTruncLogNormal,
    RandUniform,
    TruncLogNormal,
)
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


###############################
#     Tests for LogNormal     #
###############################


def test_log_normal_str() -> None:
    assert str(
        LogNormal(mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0))
    ).startswith("LogNormalSequenceGenerator(")


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_log_normal_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = LogNormal(
        mean=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        std=RandUniform(low=1.0, high=2.0, feature_size=feature_size),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_log_normal_generate_mock() -> None:
    generator = LogNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
    )
    mock = Mock(return_value=torch.ones(2, 4, 1))
    with patch("startorch.sequence.lognormal.log_normal", mock):
        generator.generate(4, 2)
        mock.assert_called_once()


def test_log_normal_generate_same_random_seed() -> None:
    generator = LogNormal(mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_log_normal_generate_different_random_seeds() -> None:
    generator = LogNormal(mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


###################################
#     Tests for RandLogNormal     #
###################################


def test_rand_log_normal_str() -> None:
    assert str(RandLogNormal()).startswith("RandLogNormalSequenceGenerator(")


@mark.parametrize("mean", (-1.0, 0.0, 1.0))
def test_rand_log_normal_mean(mean: float) -> None:
    assert RandLogNormal(mean=mean)._mean == mean


def test_rand_log_normal_mean_default() -> None:
    assert RandLogNormal()._mean == 0.0


@mark.parametrize("std", (1.0, 2.0))
def test_rand_log_normal_std(std: float) -> None:
    assert RandLogNormal(std=std)._std == std


def test_rand_log_normal_std_default() -> None:
    assert RandLogNormal()._std == 1.0


@mark.parametrize("std", (0.0, -1.0))
def test_rand_log_normal_incorrect_std(std: float) -> None:
    with raises(ValueError, match="std has to be greater than 0"):
        RandLogNormal(std=std)


def test_rand_log_normal_feature_size_default() -> None:
    assert RandLogNormal()._feature_size == (1,)


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_rand_log_normal_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandLogNormal().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_rand_log_normal_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandLogNormal(feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_rand_log_normal_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandLogNormal(feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


@mark.parametrize("mean", (1.0, 2.0))
@mark.parametrize("std", (1.0, 0.2))
def test_rand_log_normal_generate_mean_std(mean: float, std: float) -> None:
    generator = RandLogNormal(mean=mean, std=std)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.lognormal.rand_log_normal", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["size"] == (2, 3, 1)
        assert mock.call_args.kwargs["mean"] == mean
        assert mock.call_args.kwargs["std"] == std


def test_rand_log_normal_generate_same_random_seed() -> None:
    generator = RandLogNormal()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_log_normal_generate_different_random_seeds() -> None:
    generator = RandLogNormal()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


########################################
#     Tests for RandTruncLogNormal     #
########################################


def test_rand_trunc_log_normal_str() -> None:
    assert str(RandTruncLogNormal()).startswith("RandTruncLogNormalSequenceGenerator(")


@mark.parametrize("mean", (-1.0, 0.0, 1.0))
def test_rand_trunc_log_normal_mean(mean: float) -> None:
    assert RandTruncLogNormal(mean=mean)._mean == mean


def test_rand_trunc_log_normal_mean_default() -> None:
    assert RandTruncLogNormal()._mean == 0.0


@mark.parametrize("std", (1.0, 2.0))
def test_rand_trunc_log_normal_std(std: float) -> None:
    assert RandTruncLogNormal(std=std)._std == std


def test_rand_trunc_log_normal_std_default() -> None:
    assert RandTruncLogNormal()._std == 1.0


@mark.parametrize("std", (0.0, -1.0))
def test_rand_trunc_log_normal_incorrect_std(std: float) -> None:
    with raises(ValueError, match="std has to be greater than 0"):
        RandTruncLogNormal(std=std)


@mark.parametrize("min_value", (0.0, 1.0))
def test_rand_trunc_log_normal_min_value(min_value: float) -> None:
    assert RandTruncLogNormal(min_value=min_value)._min_value == min_value


def test_rand_trunc_log_normal_min_value_default() -> None:
    assert RandTruncLogNormal()._min_value == 0.0


@mark.parametrize("max_value", (1.0, 2.0))
def test_rand_trunc_log_normal_max_value(max_value: float) -> None:
    assert RandTruncLogNormal(max_value=max_value)._max_value == max_value


def test_rand_trunc_log_normal_max_value_default() -> None:
    assert RandTruncLogNormal()._max_value == 5.0


def test_rand_trunc_log_normal_incorrect_min_max_value() -> None:
    with raises(ValueError, match="max_value (.*) has to be greater or equal to min_value"):
        RandTruncLogNormal(min_value=3, max_value=2)


def test_rand_trunc_log_normal_feature_size_default() -> None:
    assert RandTruncLogNormal()._feature_size == (1,)


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_rand_trunc_log_normal_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandTruncLogNormal().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() < 5.0


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_rand_trunc_log_normal_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandTruncLogNormal(feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() < 5.0


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_rand_trunc_log_normal_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandTruncLogNormal(feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() < 5.0


@mark.parametrize("mean", (0.0, 1.0))
@mark.parametrize("std", (0.1, 1.0))
def test_rand_trunc_log_normal_generate_mean_std(mean: float, std: float) -> None:
    generator = RandTruncLogNormal(mean=mean, std=std)
    mock = Mock(return_value=torch.ones(2, 3, 1))
    with patch("startorch.sequence.lognormal.rand_trunc_log_normal", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["size"] == (2, 3, 1)
        assert mock.call_args.kwargs["mean"] == mean
        assert mock.call_args.kwargs["std"] == std


@mark.parametrize("min_value", (-2.0, -1.0))
@mark.parametrize("max_value", (2.0, 1.0))
def test_rand_trunc_log_normal_generate_min_max(min_value: float, max_value: float) -> None:
    generator = RandTruncLogNormal(min_value=min_value, max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 3, 1))
    with patch("startorch.sequence.lognormal.rand_trunc_log_normal", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["size"] == (2, 3, 1)
        assert mock.call_args.kwargs["max_value"] == max_value
        assert mock.call_args.kwargs["min_value"] == min_value


def test_rand_trunc_log_normal_generate_same_random_seed() -> None:
    generator = RandTruncLogNormal()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_trunc_log_normal_generate_different_random_seeds() -> None:
    generator = RandTruncLogNormal()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
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
    ).startswith("TruncLogNormalSequenceGenerator(")


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_trunc_log_normal_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = TruncLogNormal(
        mean=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        std=RandUniform(low=1.0, high=2.0, feature_size=feature_size),
        min_value=RandUniform(low=0.0, high=0.5, feature_size=feature_size),
        max_value=RandUniform(low=5.0, high=10.0, feature_size=feature_size),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_trunc_log_normal_generate_mock() -> None:
    generator = TruncLogNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=0.0, high=0.5),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    mock = Mock(return_value=torch.ones(2, 4, 1))
    with patch("startorch.sequence.lognormal.trunc_log_normal", mock):
        generator.generate(4, 2)
        mock.assert_called_once()


def test_trunc_log_normal_generate_same_random_seed() -> None:
    generator = TruncLogNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=0.0, high=0.5),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_trunc_log_normal_generate_different_random_seeds() -> None:
    generator = TruncLogNormal(
        mean=RandUniform(low=-1.0, high=1.0),
        std=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=0.0, high=0.5),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
