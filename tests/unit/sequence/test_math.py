from __future__ import annotations

import pytest
import torch
from objectory import OBJECT_TARGET
from redcat import BatchedTensorSeq

from startorch.sequence import (
    Abs,
    Add,
    AddScalar,
    Clamp,
    Cumsum,
    Div,
    Exp,
    Fmod,
    Full,
    Log,
    Mul,
    MulScalar,
    Neg,
    RandInt,
    RandNormal,
    RandUniform,
    Sqrt,
    Sub,
)
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


#########################
#     Tests for Abs     #
#########################


def test_abs_str() -> None:
    assert str(Abs(RandNormal())).startswith("AbsSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_abs_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    generator = Abs(RandNormal(feature_size=feature_size))
    batch = generator.generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


def test_abs_generate_fixed_value() -> None:
    assert (
        Abs(Full(-1.0))
        .generate(4, 2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0]]])
            )
        )
    )


def test_abs_generate_same_random_seed() -> None:
    generator = Abs(RandNormal())
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_abs_generate_different_random_seeds() -> None:
    generator = Abs(RandNormal())
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#########################
#     Tests for Add     #
#########################


def test_add_str() -> None:
    assert str(Add((RandUniform(), RandUniform()))).startswith("AddSequenceGenerator(")


def test_add_2_sequences() -> None:
    generator = Add((RandUniform(), {OBJECT_TARGET: "startorch.sequence.RandUniform"}))
    assert len(generator._sequences) == 2
    assert isinstance(generator._sequences[0], RandUniform)
    assert isinstance(generator._sequences[1], RandUniform)


def test_add_3_sequences() -> None:
    generator = Add(
        (
            RandUniform(),
            RandNormal(),
            {OBJECT_TARGET: "startorch.sequence.RandUniform"},
        )
    )
    assert len(generator._sequences) == 3
    assert isinstance(generator._sequences[0], RandUniform)
    assert isinstance(generator._sequences[1], RandNormal)
    assert isinstance(generator._sequences[2], RandUniform)


def test_add_sequences_empty() -> None:
    with pytest.raises(ValueError, match="No sequence generator."):
        Add(sequences=[])


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_add_generate(batch_size: int, seq_len: int) -> None:
    batch = Add((RandUniform(), RandUniform())).generate(seq_len=seq_len, batch_size=batch_size)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float


def test_add_generate_fixed_values() -> None:
    assert (
        Add((Full(1.0), Full(2.0), Full(5.0)))
        .generate(seq_len=5, batch_size=2)
        .equal(BatchedTensorSeq(torch.full((2, 5, 1), fill_value=8.0, dtype=torch.float)))
    )


def test_add_generate_same_random_seed() -> None:
    generator = Add((RandUniform(), RandUniform()))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_add_generate_different_random_seeds() -> None:
    generator = Add((RandUniform(), RandUniform()))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


###############################
#     Tests for AddScalar     #
###############################


def test_add_scalar_str() -> None:
    assert str(AddScalar(RandUniform(), value=1.0)).startswith("AddScalarSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_add_scalar_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = AddScalar(RandUniform(feature_size=feature_size), value=1.0).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 1.0
    assert batch.max() < 2.0


def test_add_scalar_generate_2() -> None:
    assert (
        AddScalar(Full(1.0), 2.0)
        .generate(4, 2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[3.0], [3.0], [3.0], [3.0]], [[3.0], [3.0], [3.0], [3.0]]])
            )
        )
    )


def test_add_scalar_generate_3() -> None:
    assert (
        AddScalar(Full(1.0), -3.0)
        .generate(4, 2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[-2.0], [-2.0], [-2.0], [-2.0]], [[-2.0], [-2.0], [-2.0], [-2.0]]])
            )
        )
    )


def test_add_scalar_generate_same_random_seed() -> None:
    generator = AddScalar(RandUniform(), value=1.0)
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_add_scalar_generate_different_random_seeds() -> None:
    generator = AddScalar(RandUniform(), value=1.0)
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


###########################
#     Tests for Clamp     #
###########################


def test_clamp_str() -> None:
    assert str(Clamp(RandNormal(), min=-2, max=2)).startswith("ClampSequenceGenerator(")


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
def test_clamp_min(min_value: float) -> None:
    assert Clamp(RandNormal(), min=min_value, max=None)._min == min_value


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_clamp_max(max_value: float) -> None:
    assert Clamp(RandNormal(), min=None, max=max_value)._max == max_value


def test_clamp_incorrect_min_max() -> None:
    with pytest.raises(ValueError, match="`min` and `max` cannot be both None"):
        Clamp(RandNormal(), min=None, max=None)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_clamp_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Clamp(RandNormal(feature_size=feature_size), min=-2, max=2).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_clamp_generate_min_max_float(min_value: float, max_value: float) -> None:
    batch = Clamp(RandNormal(), min=min_value, max=max_value).generate(batch_size=10, seq_len=10)
    assert batch.min() >= min_value
    assert batch.max() <= max_value


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_clamp_generate_min_max_long(min_value: int, max_value: int) -> None:
    batch = Clamp(RandInt(low=-5, high=20), min=min_value, max=max_value).generate(
        batch_size=10, seq_len=10
    )
    assert batch.min() >= min_value
    assert batch.max() <= max_value


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
def test_clamp_generate_only_min_value(min_value: float) -> None:
    assert (
        Clamp(RandNormal(), min=min_value, max=None).generate(batch_size=10, seq_len=10).min()
        >= min_value
    )


@pytest.mark.parametrize("max_value", [-1.0, -2.0])
def test_clamp_generate_only_max_value(max_value: float) -> None:
    assert (
        Clamp(RandNormal(), min=None, max=max_value).generate(batch_size=10, seq_len=10).max()
        <= max_value
    )


def test_clamp_generate_same_random_seed() -> None:
    generator = Clamp(RandNormal(), min=-2, max=2)
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_clamp_generate_different_random_seeds() -> None:
    generator = Clamp(RandNormal(), min=-2, max=2)
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


############################
#     Tests for Cumsum     #
############################


def test_cumsum_str() -> None:
    assert str(Cumsum(RandNormal())).startswith("CumsumSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_cumsum_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Cumsum(RandUniform(feature_size=feature_size)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_cumsum_generate_fixed_value() -> None:
    assert (
        Cumsum(Full(1.0))
        .generate(4, 2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[1.0], [2.0], [3.0], [4.0]], [[1.0], [2.0], [3.0], [4.0]]])
            )
        )
    )


def test_cumsum_generate_same_random_seed() -> None:
    generator = Cumsum(RandUniform())
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_cumsum_generate_different_random_seeds() -> None:
    generator = Cumsum(RandUniform())
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#########################
#     Tests for Div     #
#########################


def test_div_str() -> None:
    assert str(Div(RandUniform(low=0.1, high=2.0), RandUniform(low=0.1, high=2.0))).startswith(
        "DivSequenceGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_div_generate(batch_size: int, seq_len: int) -> None:
    batch = Div(
        dividend=RandUniform(low=0.1, high=2.0),
        divisor=RandUniform(low=0.1, high=2.0),
    ).generate(seq_len=seq_len, batch_size=batch_size)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_div_generate_rounding_mode_default(batch_size: int, seq_len: int) -> None:
    assert (
        Div(dividend=Full(3.0), divisor=Full(2.0))
        .generate(seq_len=seq_len, batch_size=batch_size)
        .equal(BatchedTensorSeq(torch.full((batch_size, seq_len, 1), 1.5)))
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_div_generate_rounding_mode_floor(batch_size: int, seq_len: int) -> None:
    assert (
        Div(dividend=Full(3.0), divisor=Full(2.0), rounding_mode="floor")
        .generate(seq_len=seq_len, batch_size=batch_size)
        .equal(BatchedTensorSeq(torch.ones(batch_size, seq_len, 1)))
    )


def test_div_generate_same_random_seed() -> None:
    generator = Div(
        dividend=RandUniform(low=0.1, high=2.0),
        divisor=RandUniform(low=0.1, high=2.0),
    )
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_div_generate_different_random_seeds() -> None:
    generator = Div(
        dividend=RandUniform(low=0.1, high=2.0),
        divisor=RandUniform(low=0.1, high=2.0),
    )
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#########################
#     Tests for Exp     #
#########################


def test_exp_str() -> None:
    assert str(Exp(RandUniform())).startswith("ExpSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_exp_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Exp(Full(value=0.0, feature_size=feature_size))
        .generate(batch_size=batch_size, seq_len=seq_len)
        .equal(BatchedTensorSeq(torch.ones(batch_size, seq_len, feature_size, dtype=torch.float)))
    )


def test_exp_generate_same_random_seed() -> None:
    generator = Exp(RandUniform())
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_exp_generate_different_random_seeds() -> None:
    generator = Exp(RandUniform())
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


##########################
#     Tests for Fmod     #
##########################


def test_fmod_str() -> None:
    assert str(
        Fmod(
            dividend=RandUniform(low=-100.0, high=100.0),
            divisor=RandUniform(low=1.0, high=10.0),
        )
    ).startswith("FmodSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_fmod_generate_divisor_generator(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Fmod(
            dividend=Full(5.0, feature_size=feature_size),
            divisor=Full(10.0, feature_size=feature_size),
        )
        .generate(seq_len=seq_len, batch_size=batch_size)
        .equal(
            BatchedTensorSeq(
                torch.full((batch_size, seq_len, feature_size), 5.0, dtype=torch.float)
            )
        )
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_fmod_generate_divisor_number(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Fmod(dividend=Full(5.0, feature_size=feature_size), divisor=10.0)
        .generate(seq_len=seq_len, batch_size=batch_size)
        .equal(
            BatchedTensorSeq(
                torch.full((batch_size, seq_len, feature_size), 5.0, dtype=torch.float)
            )
        )
    )


def test_fmod_generate_same_random_seed() -> None:
    generator = Fmod(
        dividend=RandUniform(low=-100.0, high=100.0), divisor=RandUniform(low=1.0, high=10.0)
    )
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_fmod_generate_different_random_seeds() -> None:
    generator = Fmod(
        dividend=RandUniform(low=-100.0, high=100.0), divisor=RandUniform(low=1.0, high=10.0)
    )
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#########################
#     Tests for Log     #
#########################


def test_log_str() -> None:
    assert str(Log(RandUniform(low=0.1, high=2.0))).startswith("LogSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_log_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Log(Full(value=1.0, feature_size=feature_size))
        .generate(batch_size=batch_size, seq_len=seq_len)
        .equal(BatchedTensorSeq(torch.zeros(batch_size, seq_len, feature_size, dtype=torch.float)))
    )


def test_log_generate_same_random_seed() -> None:
    generator = Log(RandUniform(low=0.1, high=2.0))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_log_generate_different_random_seeds() -> None:
    generator = Log(RandUniform(low=0.1, high=2.0))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#########################
#     Tests for Mul     #
#########################


def test_mul_str() -> None:
    assert str(Mul((RandUniform(), RandUniform()))).startswith("MulSequenceGenerator(")


def test_mul_2_sequences() -> None:
    generator = Mul((RandUniform(), {OBJECT_TARGET: "startorch.sequence.RandUniform"}))
    assert len(generator._sequences) == 2
    assert isinstance(generator._sequences[0], RandUniform)
    assert isinstance(generator._sequences[1], RandUniform)


def test_mul_3_sequences() -> None:
    generator = Mul(
        (
            RandUniform(),
            RandNormal(),
            {OBJECT_TARGET: "startorch.sequence.RandUniform"},
        )
    )
    assert len(generator._sequences) == 3
    assert isinstance(generator._sequences[0], RandUniform)
    assert isinstance(generator._sequences[1], RandNormal)
    assert isinstance(generator._sequences[2], RandUniform)


def test_mul_sequences_empty() -> None:
    with pytest.raises(ValueError, match="No sequence generator."):
        Mul(sequences=[])


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_mul_generate(batch_size: int, seq_len: int) -> None:
    batch = Mul((RandUniform(), RandUniform())).generate(seq_len=seq_len, batch_size=batch_size)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float


def test_mul_generate_weight() -> None:
    assert (
        Mul((Full(1.0), Full(2.0), Full(5.0)))
        .generate(seq_len=5, batch_size=2)
        .equal(BatchedTensorSeq(torch.full((2, 5, 1), fill_value=10.0, dtype=torch.float)))
    )


def test_mul_generate_same_random_seed() -> None:
    generator = Mul((RandUniform(), RandUniform()))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_mul_generate_different_random_seeds() -> None:
    generator = Mul((RandUniform(), RandUniform()))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


###############################
#     Tests for MulScalar     #
###############################


def test_mul_scalar_str() -> None:
    assert str(MulScalar(RandUniform(), value=1.0)).startswith("MulScalarSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_mul_scalar_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = MulScalar(RandUniform(feature_size=feature_size), value=2.0).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0
    assert batch.max() < 2.0


def test_mul_scalar_generate_2() -> None:
    assert (
        MulScalar(Full(1.0), 2.0)
        .generate(4, 2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[2.0], [2.0], [2.0], [2.0]], [[2.0], [2.0], [2.0], [2.0]]])
            )
        )
    )


def test_mul_scalar_generate_3() -> None:
    assert (
        MulScalar(Full(1.0), -3.0)
        .generate(4, 2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[-3.0], [-3.0], [-3.0], [-3.0]], [[-3.0], [-3.0], [-3.0], [-3.0]]])
            )
        )
    )


def test_mul_scalar_generate_same_random_seed() -> None:
    generator = MulScalar(RandUniform(), value=1.0)
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_mul_scalar_generate_different_random_seeds() -> None:
    generator = MulScalar(RandUniform(), value=1.0)
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#########################
#     Tests for Neg     #
#########################


def test_neg_str() -> None:
    assert str(Neg(RandUniform())).startswith("NegSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_neg_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    generator = Neg(RandUniform(feature_size=feature_size))
    batch = generator.generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_neg_generate_fixed_value() -> None:
    assert (
        Neg(Full(1.0))
        .generate(4, 2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[-1.0], [-1.0], [-1.0], [-1.0]], [[-1.0], [-1.0], [-1.0], [-1.0]]])
            )
        )
    )


def test_neg_generate_same_random_seed() -> None:
    generator = Neg(RandUniform())
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_neg_generate_different_random_seeds() -> None:
    generator = Neg(RandUniform())
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


##########################
#     Tests for Sqrt     #
##########################


def test_sqrt_str() -> None:
    assert str(Sqrt(RandUniform(low=0.1, high=2.0))).startswith("SqrtSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_sqrt_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Sqrt(Full(value=4.0, feature_size=feature_size))
        .generate(batch_size=batch_size, seq_len=seq_len)
        .equal(
            BatchedTensorSeq(
                torch.full((batch_size, seq_len, feature_size), 2.0, dtype=torch.float)
            )
        )
    )


def test_sqrt_generate_same_random_seed() -> None:
    generator = Sqrt(RandUniform(low=0.1, high=2.0))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_sqrt_generate_different_random_seeds() -> None:
    generator = Sqrt(RandUniform(low=0.1, high=2.0))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#########################
#     Tests for Sub     #
#########################


def test_sub_str() -> None:
    assert str(Sub(sequence1=RandUniform(), sequence2=RandUniform())).startswith(
        "SubSequenceGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_sub_generate(batch_size: int, seq_len: int) -> None:
    batch = Sub(sequence1=RandUniform(), sequence2=RandUniform()).generate(
        seq_len=seq_len, batch_size=batch_size
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float


def test_sub_generate_same_random_seed() -> None:
    generator = Sub(sequence1=RandUniform(), sequence2=RandUniform())
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_sub_generate_different_random_seeds() -> None:
    generator = Sub(sequence1=RandUniform(), sequence2=RandUniform())
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
