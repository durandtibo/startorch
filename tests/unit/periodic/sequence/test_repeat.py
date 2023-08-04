from __future__ import annotations

from unittest.mock import Mock

import torch
from pytest import mark
from redcat import BatchedTensorSeq

from startorch.periodic.sequence import Repeat
from startorch.sequence import BaseSequenceGenerator, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)
DTYPES = (torch.float, torch.long)


############################
#     Tests for Repeat     #
############################


def test_repeat_str() -> None:
    assert str(Repeat(RandUniform())).startswith("RepeatPeriodicSequenceGenerator(")


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("period", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_repeat_generate(batch_size: int, seq_len: int, period: int, feature_size: int) -> None:
    batch = Repeat(RandUniform(feature_size=feature_size)).generate(
        batch_size=batch_size, seq_len=seq_len, period=period
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_repeat_period_3() -> None:
    batch = Repeat(RandUniform()).generate(batch_size=2, period=3, seq_len=10)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == 2
    assert batch.seq_len == 10
    assert batch.data[:, :3].equal(batch.data[:, 3:6])
    assert batch.data[:, :3].equal(batch.data[:, 6:9])


def test_repeat_period_4() -> None:
    batch = Repeat(RandUniform()).generate(batch_size=2, period=4, seq_len=10)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == 2
    assert batch.seq_len == 10
    assert batch.data[:, :4].equal(batch.data[:, 4:8])
    assert batch.data[:, :2].equal(batch.data[:, 8:])


@mark.parametrize("dtype", DTYPES)
def test_repeat_dtype_dim_2(dtype: torch.dtype) -> None:
    assert (
        Repeat(
            Mock(
                spec=BaseSequenceGenerator,
                generate=Mock(return_value=BatchedTensorSeq(torch.ones(2, 3, dtype=dtype))),
            )
        )
        .generate(batch_size=2, seq_len=6, period=3)
        .equal(BatchedTensorSeq(torch.ones(2, 6, dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_repeat_dtype_dim_3(dtype: torch.dtype) -> None:
    assert (
        Repeat(
            Mock(
                spec=BaseSequenceGenerator,
                generate=Mock(return_value=BatchedTensorSeq(torch.ones(2, 3, 4, dtype=dtype))),
            )
        )
        .generate(batch_size=2, seq_len=6, period=3)
        .equal(BatchedTensorSeq(torch.ones(2, 6, 4, dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_repeat_dtype_dim_4(dtype: torch.dtype) -> None:
    assert (
        Repeat(
            Mock(
                spec=BaseSequenceGenerator,
                generate=Mock(return_value=BatchedTensorSeq(torch.ones(2, 3, 4, 5, dtype=dtype))),
            )
        )
        .generate(batch_size=2, seq_len=6, period=3)
        .equal(BatchedTensorSeq(torch.ones(2, 6, 4, 5, dtype=dtype)))
    )


def test_repeat_generate_same_random_seed() -> None:
    generator = Repeat(RandUniform())
    assert generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1))
    )


def test_repeat_generate_different_random_seeds() -> None:
    generator = Repeat(RandUniform())
    assert not generator.generate(
        batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)
    ).equal(generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(2)))
