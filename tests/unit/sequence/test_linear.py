from __future__ import annotations

import torch
from pytest import mark
from redcat import BatchedTensorSeq

from startorch.sequence import Full, Linear, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


############################
#     Tests for Linear     #
############################


def test_linear_str() -> None:
    assert str(
        Linear(
            value=RandUniform(low=-1.0, high=1.0),
            slope=RandUniform(low=-1.0, high=1.0),
            intercept=RandUniform(low=-1.0, high=1.0),
        )
    ).startswith("LinearSequenceGenerator(")


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_linear_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Linear(
        value=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        slope=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        intercept=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_linear_generate_fixed() -> None:
    assert (
        Linear(value=Full(2.0), slope=Full(-3.0), intercept=Full(1.0))
        .generate(batch_size=2, seq_len=4)
        .equal(BatchedTensorSeq(torch.full((2, 4, 1), -5.0)))
    )


def test_linear_generate_same_random_seed() -> None:
    generator = Linear(
        value=RandUniform(low=-1.0, high=1.0),
        slope=RandUniform(low=1.0, high=2.0),
        intercept=RandUniform(low=-10.0, high=-5.0),
    )
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_linear_generate_different_random_seeds() -> None:
    generator = Linear(
        value=RandUniform(low=-1.0, high=1.0),
        slope=RandUniform(low=1.0, high=2.0),
        intercept=RandUniform(low=-10.0, high=-5.0),
    )
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
