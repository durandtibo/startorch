from __future__ import annotations

from unittest.mock import Mock

import torch
from pytest import mark
from redcat import BatchDict, BatchedTensorSeq

from startorch import constants as ct
from startorch.periodic.timeseries import BasePeriodicTimeSeriesGenerator, Repeat
from startorch.sequence import RandUniform
from startorch.tensor import BaseTensorGenerator, RandInt
from startorch.timeseries import BaseTimeSeriesGenerator, Periodic, TimeSeries
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2)


##############################
#     Tests for Periodic     #
##############################


def test_periodic_str() -> None:
    assert str(
        Periodic(
            timeseries=TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}),
            period=RandInt(2, 5),
        )
    ).startswith("PeriodicTimeSeriesGenerator(")


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize(
    "timeseries",
    (
        TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}),
        Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})),
    ),
)
def test_periodic_generate(
    batch_size: int,
    seq_len: int,
    timeseries: BaseTimeSeriesGenerator | BasePeriodicTimeSeriesGenerator,
) -> None:
    batch = Periodic(
        timeseries=timeseries,
        period=RandInt(2, 5),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchDict)
    assert batch.batch_size == batch_size
    assert len(batch.data) == 2

    batch_value = batch.data[ct.VALUE]
    assert isinstance(batch_value, BatchedTensorSeq)
    assert batch_value.batch_size == batch_size
    assert batch_value.seq_len == seq_len
    assert batch_value.data.shape == (batch_size, seq_len, 1)
    assert batch_value.data.dtype == torch.float

    batch_time = batch.data[ct.TIME]
    assert isinstance(batch_time, BatchedTensorSeq)
    assert batch_time.batch_size == batch_size
    assert batch_time.seq_len == seq_len
    assert batch_time.data.shape == (batch_size, seq_len, 1)
    assert batch_time.data.dtype == torch.float


@mark.parametrize(
    "timeseries",
    (
        TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}),
        Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})),
    ),
)
def test_periodic_generate_period_4(
    timeseries: BaseTimeSeriesGenerator | BasePeriodicTimeSeriesGenerator,
) -> None:
    batch = Periodic(
        timeseries=timeseries,
        period=Mock(spec=BaseTensorGenerator, generate=Mock(return_value=torch.tensor([4]))),
    ).generate(batch_size=2, seq_len=10)
    assert isinstance(batch, BatchDict)
    assert batch.batch_size == 2
    assert batch.slice_along_seq(0, 4).equal(batch.slice_along_seq(4, 8))
    assert batch.slice_along_seq(0, 2).equal(batch.slice_along_seq(8))


def test_periodic_generate_same_random_seed() -> None:
    generator = Periodic(
        timeseries=TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}),
        period=RandInt(2, 5),
    )
    assert generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1)).equal(
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1))
    )


def test_periodic_generate_different_random_seeds() -> None:
    generator = Periodic(
        timeseries=TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}),
        period=RandInt(2, 5),
    )
    assert not generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1)).equal(
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(2))
    )
