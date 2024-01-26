from __future__ import annotations

import pytest
import torch
from redcat import BatchedTensorSeq

from startorch.sequence import Float, Full, Long, RandInt, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


###########################
#     Tests for Float     #
###########################


def test_float_str() -> None:
    assert str(Float(RandInt(low=0, high=10))).startswith("FloatSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_float_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Float(Full(value=4, feature_size=feature_size))
        .generate(batch_size=batch_size, seq_len=seq_len)
        .equal(
            BatchedTensorSeq(
                torch.full((batch_size, seq_len, feature_size), 4.0, dtype=torch.float)
            )
        )
    )


def test_float_generate_same_random_seed() -> None:
    generator = Float(RandInt(low=0, high=10))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_float_generate_different_random_seeds() -> None:
    generator = Float(RandInt(low=0, high=10))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


##########################
#     Tests for Long     #
##########################


def test_long_str() -> None:
    assert str(Long(RandUniform(low=0.0, high=50.0))).startswith("LongSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_long_generate(batch_size: int, seq_len: int) -> None:
    batch = Long(RandUniform(low=0.0, high=50.0)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.long


def test_long_generate_same_random_seed() -> None:
    generator = Long(RandUniform(low=0.0, high=50.0))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_long_generate_different_random_seeds() -> None:
    generator = Long(RandUniform(low=0.0, high=50.0))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
