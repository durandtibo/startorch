from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from startorch.sequence import (
    HalfCauchy,
    RandHalfCauchy,
    RandTruncHalfCauchy,
    RandUniform,
    TruncHalfCauchy,
)
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


################################
#     Tests for HalfCauchy     #
################################


def test_half_cauchy_str() -> None:
    assert str(HalfCauchy(scale=RandUniform(low=1.0, high=2.0))).startswith(
        "HalfCauchySequenceGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_half_cauchy_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = HalfCauchy(scale=RandUniform(low=1.0, high=2.0, feature_size=feature_size)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


def test_half_cauchy_generate_mock() -> None:
    generator = HalfCauchy(scale=RandUniform(low=1.0, high=2.0))
    mock = Mock(return_value=torch.ones(2, 4, 1))
    with patch("startorch.sequence.halfcauchy.half_cauchy", mock):
        generator.generate(4, 2)
        mock.assert_called_once()


def test_half_cauchy_generate_same_random_seed() -> None:
    generator = HalfCauchy(RandUniform(low=1.0, high=2.0))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_half_cauchy_generate_different_random_seeds() -> None:
    generator = HalfCauchy(RandUniform(low=1.0, high=2.0))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


####################################
#     Tests for RandHalfCauchy     #
####################################


def test_rand_half_cauchy_str() -> None:
    assert str(RandHalfCauchy()).startswith("RandHalfCauchySequenceGenerator(")


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rand_half_cauchy_scale(scale: float) -> None:
    assert RandHalfCauchy(scale=scale)._scale == scale


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_rand_half_cauchy_incorrect_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="scale has to be greater than 0"):
        RandHalfCauchy(scale=scale)


def test_rand_half_cauchy_feature_size_default() -> None:
    assert RandHalfCauchy()._feature_size == (1,)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_half_cauchy_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandHalfCauchy().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 1)
    assert batch.dtype == torch.float
    assert batch.min() >= 0.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_rand_half_cauchy_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandHalfCauchy(feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float
    assert batch.min() >= 0.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_half_cauchy_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandHalfCauchy(feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)

    assert batch.shape == (batch_size, seq_len, 3, 4)
    assert batch.dtype == torch.float
    assert batch.min() >= 0.0


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rand_half_cauchy_generate_scale(scale: float) -> None:
    generator = RandHalfCauchy(scale=scale)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.halfcauchy.rand_half_cauchy", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["scale"] == scale


def test_rand_half_cauchy_generate_same_random_seed() -> None:
    generator = RandHalfCauchy()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_half_cauchy_generate_different_random_seeds() -> None:
    generator = RandHalfCauchy()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#########################################
#     Tests for RandTruncHalfCauchy     #
#########################################


def test_rand_trunc_half_cauchy_str() -> None:
    assert str(RandTruncHalfCauchy()).startswith("RandTruncHalfCauchySequenceGenerator(")


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


def test_rand_trunc_half_cauchy_feature_size_default() -> None:
    assert RandTruncHalfCauchy()._feature_size == (1,)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_trunc_half_cauchy_generate_feature_size_default(
    batch_size: int, seq_len: int
) -> None:
    batch = RandTruncHalfCauchy().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 1)
    assert batch.dtype == torch.float
    assert batch.min() >= 0
    assert batch.max() <= 4


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_rand_trunc_half_cauchy_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandTruncHalfCauchy(feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float
    assert batch.min() >= 0
    assert batch.max() <= 4


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_trunc_half_cauchy_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandTruncHalfCauchy(feature_size=(3, 4)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 3, 4)
    assert batch.dtype == torch.float
    assert batch.min() >= 0
    assert batch.max() <= 4


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rand_trunc_half_cauchy_generate_scale(scale: float) -> None:
    generator = RandTruncHalfCauchy(scale=scale)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.halfcauchy.rand_trunc_half_cauchy", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["scale"] == scale


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_rand_trunc_half_cauchy_generate_max_value(max_value: float) -> None:
    generator = RandTruncHalfCauchy(max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.halfcauchy.rand_trunc_half_cauchy", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["max_value"] == max_value


def test_rand_trunc_half_cauchy_generate_same_random_seed() -> None:
    generator = RandTruncHalfCauchy()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_trunc_half_cauchy_generate_different_random_seeds() -> None:
    generator = RandTruncHalfCauchy()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
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
    ).startswith("TruncHalfCauchySequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_trunc_half_cauchy_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = TruncHalfCauchy(
        scale=RandUniform(low=1.0, high=2.0, feature_size=feature_size),
        max_value=RandUniform(low=5.0, high=10.0, feature_size=feature_size),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


def test_trunc_half_cauchy_generate_mock() -> None:
    generator = TruncHalfCauchy(
        scale=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    mock = Mock(return_value=torch.ones(2, 4, 1))
    with patch("startorch.sequence.halfcauchy.trunc_half_cauchy", mock):
        generator.generate(4, 2)
        mock.assert_called_once()


def test_trunc_half_cauchy_generate_same_random_seed() -> None:
    generator = TruncHalfCauchy(
        scale=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_trunc_half_cauchy_generate_different_random_seeds() -> None:
    generator = TruncHalfCauchy(
        scale=RandUniform(low=1.0, high=2.0), max_value=RandUniform(low=5.0, high=10.0)
    )
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
