from unittest.mock import Mock, patch

import pytest
import torch
from redcat import BatchedTensorSeq

from startorch.sequence import (
    HalfNormal,
    RandHalfNormal,
    RandTruncHalfNormal,
    RandUniform,
    TruncHalfNormal,
)
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


################################
#     Tests for HalfNormal     #
################################


def test_half_normal_str() -> None:
    assert str(HalfNormal(std=RandUniform(low=1.0, high=2.0))).startswith(
        "HalfNormalSequenceGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_half_normal_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = HalfNormal(std=RandUniform(low=1.0, high=2.0, feature_size=feature_size)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_half_normal_generate_mock() -> None:
    generator = HalfNormal(std=RandUniform(low=1.0, high=2.0))
    mock = Mock(return_value=torch.ones(2, 4, 1))
    with patch("startorch.sequence.halfnormal.half_normal", mock):
        generator.generate(4, 2)
        mock.assert_called_once()


def test_half_normal_generate_same_random_seed() -> None:
    generator = HalfNormal(RandUniform(low=1.0, high=2.0))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_half_normal_generate_different_random_seeds() -> None:
    generator = HalfNormal(RandUniform(low=1.0, high=2.0))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


####################################
#     Tests for RandHalfNormal     #
####################################


def test_rand_half_normal_str() -> None:
    assert str(RandHalfNormal()).startswith("RandHalfNormalSequenceGenerator(")


@pytest.mark.parametrize("std", [1, 2])
def test_rand_half_normal_std(std: float) -> None:
    assert RandHalfNormal(std=std)._std == std


def test_rand_half_normal_std_default() -> None:
    assert RandHalfNormal()._std == 1.0


@pytest.mark.parametrize("std", [0, -1])
def test_rand_half_normal_incorrect_std(std: float) -> None:
    with pytest.raises(ValueError, match="std has to be greater than 0"):
        RandHalfNormal(std=std)


def test_rand_half_normal_feature_size_default() -> None:
    assert RandHalfNormal()._feature_size == (1,)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_half_normal_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandHalfNormal().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_rand_half_normal_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandHalfNormal(feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_half_normal_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandHalfNormal(feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


@pytest.mark.parametrize("std", [0.1, 1.0])
def test_rand_half_normal_generate_mean_std(std: float) -> None:
    generator = RandHalfNormal(std=std)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.halfnormal.rand_half_normal", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["std"] == std


def test_rand_half_normal_generate_same_random_seed() -> None:
    generator = RandHalfNormal()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_half_normal_generate_different_random_seeds() -> None:
    generator = RandHalfNormal()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#########################################
#     Tests for RandTruncHalfNormal     #
#########################################


def test_rand_trunc_normal_str() -> None:
    assert str(RandTruncHalfNormal()).startswith("RandTruncHalfNormalSequenceGenerator(")


@pytest.mark.parametrize("std", [1.0, 2.0])
def test_rand_trunc_normal_std(std: float) -> None:
    assert RandTruncHalfNormal(std=std)._std == std


def test_rand_trunc_normal_std_default() -> None:
    assert RandTruncHalfNormal()._std == 1.0


@pytest.mark.parametrize("std", [0.0, -1.0])
def test_rand_trunc_normal_incorrect_std(std: float) -> None:
    with pytest.raises(ValueError, match="std has to be greater than 0"):
        RandTruncHalfNormal(std=std)


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_rand_trunc_normal_max_value(max_value: float) -> None:
    assert RandTruncHalfNormal(max_value=max_value)._max_value == max_value


def test_rand_trunc_normal_max_value_default() -> None:
    assert RandTruncHalfNormal()._max_value == 3.0


@pytest.mark.parametrize("max_value", [0.0, -1.0])
def test_rand_trunc_normal_incorrect_max_value(max_value: float) -> None:
    with pytest.raises(ValueError, match="max_value has to be greater than 0"):
        RandTruncHalfNormal(max_value=max_value)


def test_rand_trunc_normal_feature_size_default() -> None:
    assert RandTruncHalfNormal()._feature_size == (1,)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_trunc_normal_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandTruncHalfNormal().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() < 3.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_rand_trunc_normal_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandTruncHalfNormal(feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() < 3.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_trunc_normal_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandTruncHalfNormal(feature_size=(3, 4)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() < 3.0


@pytest.mark.parametrize("std", [0.1, 1.0])
def test_rand_trunc_normal_generate_std(std: float) -> None:
    generator = RandTruncHalfNormal(std=std)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.halfnormal.rand_trunc_half_normal", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["std"] == std


@pytest.mark.parametrize("max_value", [2.0, 1.0])
def test_rand_trunc_normal_generate_min_max(max_value: float) -> None:
    generator = RandTruncHalfNormal(max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.halfnormal.rand_trunc_half_normal", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["max_value"] == max_value


def test_rand_trunc_normal_generate_same_random_seed() -> None:
    generator = RandTruncHalfNormal()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_trunc_normal_generate_different_random_seeds() -> None:
    generator = RandTruncHalfNormal()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
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
    ).startswith("TruncHalfNormalSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_trunc_half_normal_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = TruncHalfNormal(
        std=RandUniform(low=1.0, high=2.0, feature_size=feature_size),
        max_value=RandUniform(low=5.0, high=10.0, feature_size=feature_size),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_trunc_half_normal_generate_mock() -> None:
    generator = TruncHalfNormal(
        std=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    mock = Mock(return_value=torch.ones(2, 4, 1))
    with patch("startorch.sequence.halfnormal.trunc_half_normal", mock):
        generator.generate(4, 2)
        mock.assert_called_once()


def test_trunc_half_normal_generate_same_random_seed() -> None:
    generator = TruncHalfNormal(
        std=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_trunc_half_normal_generate_different_random_seeds() -> None:
    generator = TruncHalfNormal(
        std=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
