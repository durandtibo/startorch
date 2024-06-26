from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch
from coola import objects_are_equal

from startorch.sequence import (
    Cauchy,
    RandCauchy,
    RandTruncCauchy,
    RandUniform,
    TruncCauchy,
)
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


############################
#     Tests for Cauchy     #
############################


def test_cauchy_str() -> None:
    assert str(
        Cauchy(loc=RandUniform(low=-1.0, high=1.0), scale=RandUniform(low=1.0, high=2.0))
    ).startswith("CauchySequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_cauchy_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Cauchy(
        loc=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        scale=RandUniform(low=1.0, high=2.0, feature_size=feature_size),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


def test_cauchy_generate_mock() -> None:
    generator = Cauchy(loc=RandUniform(low=-1.0, high=1.0), scale=RandUniform(low=1.0, high=2.0))
    mock = Mock(return_value=torch.ones(2, 4, 1))
    with patch("startorch.sequence.cauchy.cauchy", mock):
        generator.generate(4, 2)
        mock.assert_called_once()


def test_cauchy_generate_same_random_seed() -> None:
    generator = Cauchy(loc=RandUniform(low=-1.0, high=1.0), scale=RandUniform(low=1.0, high=2.0))
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_cauchy_generate_different_random_seeds() -> None:
    generator = Cauchy(loc=RandUniform(low=-1.0, high=1.0), scale=RandUniform(low=1.0, high=2.0))
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )


################################
#     Tests for RandCauchy     #
################################


def test_rand_cauchy_str() -> None:
    assert str(RandCauchy()).startswith("RandCauchySequenceGenerator(")


@pytest.mark.parametrize("loc", [-1.0, 0.0, 1.0])
def test_rand_cauchy_loc(loc: float) -> None:
    assert RandCauchy(loc=loc)._loc == loc


def test_rand_cauchy_loc_default() -> None:
    assert RandCauchy()._loc == 0.0


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rand_cauchy_scale(scale: float) -> None:
    assert RandCauchy(scale=scale)._scale == scale


def test_rand_cauchy_scale_default() -> None:
    assert RandCauchy()._scale == 1.0


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_rand_cauchy_incorrect_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="scale has to be greater than 0"):
        RandCauchy(scale=scale)


def test_rand_cauchy_feature_size_default() -> None:
    assert RandCauchy()._feature_size == (1,)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_cauchy_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandCauchy().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 1)
    assert batch.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_rand_cauchy_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandCauchy(feature_size=feature_size).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_cauchy_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandCauchy(feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 3, 4)
    assert batch.dtype == torch.float


@pytest.mark.parametrize("loc", [0, 1])
@pytest.mark.parametrize("scale", [1, 2])
def test_rand_cauchy_generate_loc_scale(loc: float, scale: float) -> None:
    generator = RandCauchy(loc=loc, scale=scale)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.cauchy.rand_cauchy", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["loc"] == loc
        assert mock.call_args.kwargs["scale"] == scale


def test_rand_cauchy_generate_same_random_seed() -> None:
    generator = RandCauchy()
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_rand_cauchy_generate_different_random_seeds() -> None:
    generator = RandCauchy()
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )


#####################################
#     Tests for RandTruncCauchy     #
#####################################


def test_rand_trunc_cauchy_str() -> None:
    assert str(RandTruncCauchy()).startswith("RandTruncCauchySequenceGenerator(")


@pytest.mark.parametrize("loc", [-1.0, 0.0, 1.0])
def test_rand_trunc_cauchy_loc(loc: float) -> None:
    assert RandTruncCauchy(loc=loc)._loc == loc


def test_rand_trunc_cauchy_loc_default() -> None:
    assert RandTruncCauchy()._loc == 0.0


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rand_trunc_cauchy_scale(scale: float) -> None:
    assert RandTruncCauchy(scale=scale)._scale == scale


def test_rand_trunc_cauchy_scale_default() -> None:
    assert RandTruncCauchy()._scale == 1.0


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_rand_trunc_cauchy_incorrect_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="scale has to be greater than 0"):
        RandTruncCauchy(scale=scale)


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
def test_rand_trunc_cauchy_min_value(min_value: float) -> None:
    assert RandTruncCauchy(min_value=min_value)._min_value == min_value


def test_rand_trunc_cauchy_min_value_default() -> None:
    assert RandTruncCauchy()._min_value == -2.0


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_rand_trunc_cauchy_max_value(max_value: float) -> None:
    assert RandTruncCauchy(max_value=max_value)._max_value == max_value


def test_rand_trunc_cauchy_max_value_default() -> None:
    assert RandTruncCauchy()._max_value == 2.0


def test_rand_trunc_cauchy_incorrect_min_max() -> None:
    with pytest.raises(ValueError, match="max_value (.*) has to be greater or equal to min_value"):
        RandTruncCauchy(min_value=1.0, max_value=-1.0)


def test_rand_trunc_cauchy_feature_size_default() -> None:
    assert RandTruncCauchy()._feature_size == (1,)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_trunc_cauchy_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandTruncCauchy().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 1)
    assert batch.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_rand_trunc_cauchy_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandTruncCauchy(feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_trunc_cauchy_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandTruncCauchy(feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 3, 4)
    assert batch.dtype == torch.float


@pytest.mark.parametrize("loc", [0, 1])
@pytest.mark.parametrize("scale", [1, 2])
def test_rand_trunc_cauchy_generate_loc_scale(loc: float, scale: float) -> None:
    generator = RandTruncCauchy(loc=loc, scale=scale)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.cauchy.rand_trunc_cauchy", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["loc"] == loc
        assert mock.call_args.kwargs["scale"] == scale


@pytest.mark.parametrize("max_value", [1.0, 2.0])
@pytest.mark.parametrize("min_value", [-1.0, -2.0])
def test_rand_trunc_cauchy_generate_max_min(max_value: float, min_value: float) -> None:
    generator = RandTruncCauchy(min_value=min_value, max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 3))
    with patch("startorch.sequence.cauchy.rand_trunc_cauchy", mock):
        generator.generate(batch_size=2, seq_len=3)
        assert mock.call_args.kwargs["max_value"] == max_value
        assert mock.call_args.kwargs["min_value"] == min_value


def test_rand_trunc_cauchy_generate_same_random_seed() -> None:
    generator = RandTruncCauchy()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_trunc_cauchy_generate_different_random_seeds() -> None:
    generator = RandTruncCauchy()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#################################
#     Tests for TruncCauchy     #
#################################


def test_trunc_cauchy_str() -> None:
    assert str(
        TruncCauchy(
            loc=RandUniform(low=-1.0, high=1.0),
            scale=RandUniform(low=1.0, high=2.0),
            min_value=RandUniform(low=-10.0, high=-5.0),
            max_value=RandUniform(low=5.0, high=10.0),
        )
    ).startswith("TruncCauchySequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_trunc_cauchy_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = TruncCauchy(
        loc=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        scale=RandUniform(low=1.0, high=2.0, feature_size=feature_size),
        min_value=RandUniform(low=-10.0, high=-5.0, feature_size=feature_size),
        max_value=RandUniform(low=5.0, high=10.0, feature_size=feature_size),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


def test_trunc_cauchy_generate_mock() -> None:
    generator = TruncCauchy(
        loc=RandUniform(low=-1.0, high=1.0),
        scale=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    mock = Mock(return_value=torch.ones(2, 4, 1))
    with patch("startorch.sequence.cauchy.trunc_cauchy", mock):
        generator.generate(4, 2)
        mock.assert_called_once()


def test_trunc_cauchy_generate_same_random_seed() -> None:
    generator = TruncCauchy(
        loc=RandUniform(low=-1.0, high=1.0),
        scale=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_trunc_cauchy_generate_different_random_seeds() -> None:
    generator = TruncCauchy(
        loc=RandUniform(low=-1.0, high=1.0),
        scale=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
