from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from batchtensor.nested import slice_along_seq
from coola import objects_are_equal

from startorch import constants as ct
from startorch.periodic.timeseries import BasePeriodicTimeSeriesGenerator, Repeat
from startorch.sequence import RandUniform
from startorch.tensor import BaseTensorGenerator, RandInt
from startorch.timeseries import BaseTimeSeriesGenerator, Periodic, SequenceTimeSeries
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


##############################
#     Tests for Periodic     #
##############################


def test_periodic_str() -> None:
    assert str(
        Periodic(
            timeseries=SequenceTimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}),
            period=RandInt(2, 5),
        )
    ).startswith("PeriodicTimeSeriesGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize(
    "timeseries",
    [
        SequenceTimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}),
        Repeat(SequenceTimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})),
    ],
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
    assert isinstance(batch, dict)
    assert len(batch) == 2

    batch_value = batch[ct.VALUE]
    assert isinstance(batch_value, torch.Tensor)
    assert batch_value.shape == (batch_size, seq_len, 1)
    assert batch_value.dtype == torch.float

    batch_time = batch[ct.TIME]
    assert isinstance(batch_time, torch.Tensor)
    assert batch_time.shape == (batch_size, seq_len, 1)
    assert batch_time.dtype == torch.float


@pytest.mark.parametrize(
    "timeseries",
    [
        SequenceTimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}),
        Repeat(SequenceTimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})),
    ],
)
def test_periodic_generate_period_4(
    timeseries: BaseTimeSeriesGenerator | BasePeriodicTimeSeriesGenerator,
) -> None:
    batch = Periodic(
        timeseries=timeseries,
        period=Mock(spec=BaseTensorGenerator, generate=Mock(return_value=torch.tensor([4]))),
    ).generate(batch_size=2, seq_len=10)
    assert isinstance(batch, dict)
    assert objects_are_equal(
        slice_along_seq(batch, start=0, stop=4), slice_along_seq(batch, start=4, stop=8)
    )
    assert objects_are_equal(
        slice_along_seq(batch, start=0, stop=2), slice_along_seq(batch, start=8)
    )


def test_periodic_generate_same_random_seed() -> None:
    generator = Periodic(
        timeseries=SequenceTimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}),
        period=RandInt(2, 5),
    )
    assert objects_are_equal(
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1)),
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1)),
    )


def test_periodic_generate_different_random_seeds() -> None:
    generator = Periodic(
        timeseries=SequenceTimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}),
        period=RandInt(2, 5),
    )
    assert not objects_are_equal(
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1)),
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(2)),
    )
