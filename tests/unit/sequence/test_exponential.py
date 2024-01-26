from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch
from redcat import BatchedTensorSeq

from startorch.sequence import (
    Exponential,
    RandExponential,
    RandTruncExponential,
    RandUniform,
    TruncExponential,
)
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


#################################
#     Tests for Exponential     #
#################################


def test_exponential_str() -> None:
    assert str(Exponential(RandUniform(low=1.0, high=5.0))).startswith(
        "ExponentialSequenceGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_exponential_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Exponential(rate=RandUniform(low=1.0, high=5.0, feature_size=feature_size)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_exponential_generate_mock() -> None:
    generator = Exponential(rate=RandUniform(low=1.0, high=5.0))
    mock = Mock(return_value=torch.ones(2, 4, 1))
    with patch("startorch.sequence.exponential.exponential", mock):
        generator.generate(4, 2)
        mock.assert_called_once()


def test_exponential_generate_same_random_seed() -> None:
    generator = Exponential(rate=RandUniform(low=1.0, high=5.0))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_exponential_generate_different_random_seeds() -> None:
    generator = Exponential(rate=RandUniform(low=1.0, high=5.0))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


@pytest.mark.parametrize(
    "generator",
    [Exponential.create_fixed_rate(), Exponential.create_uniform_rate()],
)
def test_exponential_generate_predefined_generators(generator: Exponential) -> None:
    batch = generator.generate(batch_size=4, seq_len=12)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == 4
    assert batch.seq_len == 12
    assert batch.data.shape == (4, 12, 1)
    assert batch.data.dtype == torch.float


#####################################
#     Tests for RandExponential     #
#####################################


def test_rand_exponential_str() -> None:
    assert str(RandExponential()).startswith("RandExponentialSequenceGenerator(")


@pytest.mark.parametrize("rate", [1.0, 2.0])
def test_rand_exponential_rate(rate: float) -> None:
    assert RandExponential(rate=rate)._rate == rate


def test_rand_exponential_rate_default() -> None:
    assert RandExponential()._rate == 1.0


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_rand_exponential_incorrect_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="rate has to be greater than 0"):
        RandExponential(rate=rate)


def test_rand_exponential_feature_size_default() -> None:
    assert RandExponential()._feature_size == (1,)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_exponential_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandExponential().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_rand_exponential_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandExponential(feature_size=feature_size).generate(
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
def test_rand_exponential_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandExponential(feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


@pytest.mark.parametrize("rate", [1, 2])
def test_rand_exponential_generate_rate(rate: float) -> None:
    generator = RandExponential(rate=rate)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.exponential.rand_exponential", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["rate"] == rate


def test_rand_exponential_generate_same_random_seed() -> None:
    generator = RandExponential()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_exponential_generate_different_random_seeds() -> None:
    generator = RandExponential()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


##########################################
#     Tests for RandTruncExponential     #
##########################################


def test_rand_trunc_exponential_str() -> None:
    assert str(RandTruncExponential()).startswith("RandTruncExponentialSequenceGenerator(")


@pytest.mark.parametrize("rate", [1.0, 2.0])
def test_rand_trunc_exponential_rate(rate: float) -> None:
    assert RandTruncExponential(rate=rate)._rate == rate


def test_rand_trunc_exponential_rate_default() -> None:
    assert RandTruncExponential()._rate == 1.0


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_rand_trunc_exponential_incorrect_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="rate has to be greater than 0"):
        RandTruncExponential(rate=rate)


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_rand_trunc_exponential_max_value(max_value: float) -> None:
    assert RandTruncExponential(max_value=max_value)._max_value == max_value


def test_rand_trunc_exponential_max_value_default() -> None:
    assert RandTruncExponential()._max_value == 5.0


@pytest.mark.parametrize("max_value", [0.0, -1.0])
def test_rand_trunc_exponential_incorrect_max_value(max_value: float) -> None:
    with pytest.raises(ValueError, match="max_value has to be greater than 0"):
        RandTruncExponential(max_value=max_value)


def test_rand_trunc_exponential_feature_size_default() -> None:
    assert RandTruncExponential()._feature_size == (1,)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_trunc_exponential_generate_feature_size_default(
    batch_size: int, seq_len: int
) -> None:
    batch = RandTruncExponential().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() <= 5.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_rand_trunc_exponential_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandTruncExponential(feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() <= 5.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_trunc_exponential_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandTruncExponential(feature_size=(3, 4)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() <= 5.0


@pytest.mark.parametrize("rate", [1, 2])
def test_rand_trunc_exponential_generate_rate(rate: float) -> None:
    generator = RandTruncExponential(rate=rate)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.exponential.rand_trunc_exponential", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["rate"] == rate


@pytest.mark.parametrize("max_value", [1, 2])
def test_rand_trunc_exponential_generate_max_value(max_value: float) -> None:
    generator = RandTruncExponential(max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.exponential.rand_trunc_exponential", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["max_value"] == max_value


def test_rand_trunc_exponential_generate_same_random_seed() -> None:
    generator = RandTruncExponential()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_trunc_exponential_generate_different_random_seeds() -> None:
    generator = RandTruncExponential()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
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
    ).startswith("TruncExponentialSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_trunc_exponential_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = TruncExponential(
        rate=RandUniform(low=1.0, high=2.0, feature_size=feature_size),
        max_value=RandUniform(low=5.0, high=10.0, feature_size=feature_size),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() <= 10.0


def test_trunc_exponential_generate_mock() -> None:
    generator = TruncExponential(
        rate=RandUniform(low=1.0, high=5.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    mock = Mock(return_value=torch.ones(2, 4, 1))
    with patch("startorch.sequence.exponential.trunc_exponential", mock):
        generator.generate(4, 2)
        mock.assert_called_once()


def test_trunc_exponential_generate_same_random_seed() -> None:
    generator = TruncExponential(
        rate=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_trunc_exponential_generate_different_random_seeds() -> None:
    generator = TruncExponential(
        rate=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
