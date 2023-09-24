from __future__ import annotations

import torch
from pytest import mark
from redcat import BatchDict, BatchedTensorSeq

from startorch import constants as ct
from startorch.periodic.timeseries import Repeat
from startorch.sequence import RandUniform
from startorch.timeseries import TimeSeries
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)
DTYPES = (torch.float, torch.long)


############################
#     Tests for Repeat     #
############################


def test_repeat_str() -> None:
    assert str(Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}))).startswith(
        "RepeatPeriodicTimeSeriesGenerator("
    )


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("period", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_repeat_generate(batch_size: int, seq_len: int, period: int, feature_size: int) -> None:
    batch = Repeat(
        TimeSeries({ct.VALUE: RandUniform(feature_size=feature_size), ct.TIME: RandUniform()})
    ).generate(batch_size=batch_size, seq_len=seq_len, period=period)

    assert isinstance(batch, BatchDict)
    assert batch.batch_size == batch_size
    assert len(batch.data) == 2

    batch_value = batch.data[ct.VALUE]
    assert isinstance(batch_value, BatchedTensorSeq)
    assert batch_value.batch_size == batch_size
    assert batch_value.seq_len == seq_len
    assert batch_value.data.shape == (batch_size, seq_len, feature_size)
    assert batch_value.data.dtype == torch.float

    batch_time = batch.data[ct.TIME]
    assert isinstance(batch_time, BatchedTensorSeq)
    assert batch_time.batch_size == batch_size
    assert batch_time.seq_len == seq_len
    assert batch_time.data.shape == (batch_size, seq_len, 1)
    assert batch_time.data.dtype == torch.float


def test_repeat_period_3() -> None:
    batch = Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})).generate(
        batch_size=2, period=3, seq_len=10
    )
    assert isinstance(batch, BatchDict)
    assert batch.batch_size == 2
    assert batch.slice_along_seq(start=0, stop=3).equal(batch.slice_along_seq(start=3, stop=6))
    assert batch.slice_along_seq(start=0, stop=3).equal(batch.slice_along_seq(start=6, stop=9))


def test_repeat_period_4() -> None:
    batch = Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})).generate(
        batch_size=2, period=4, seq_len=10
    )
    assert isinstance(batch, BatchDict)
    assert batch.batch_size == 2
    assert batch.slice_along_seq(start=0, stop=4).equal(batch.slice_along_seq(start=4, stop=8))
    assert batch.slice_along_seq(start=0, stop=2).equal(batch.slice_along_seq(start=8, stop=10))


def test_repeat_generate_same_random_seed() -> None:
    generator = Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}))
    assert generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1))
    )


def test_repeat_generate_different_random_seeds() -> None:
    generator = Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}))
    assert not generator.generate(
        batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)
    ).equal(generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(2)))
