from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch import timeseries
from startorch.example import TimeSeries
from startorch.sequence import RandUniform
from startorch.tensor import Full, RandInt
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


################################################
#     Tests for TimeSeriesExampleGenerator     #
################################################


def test_timeseries_str() -> None:
    assert str(
        TimeSeries(
            timeseries=timeseries.TimeSeries({"value": RandUniform(), "time": RandUniform()}),
            seq_len=RandInt(2, 5),
        )
    ).startswith("TimeSeriesExampleGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_timeseries_generate(batch_size: int, seq_len: int) -> None:
    batch = TimeSeries(
        timeseries=timeseries.TimeSeries({"value": RandUniform(), "time": RandUniform()}),
        seq_len=Full(seq_len),
    ).generate(batch_size=batch_size)
    assert isinstance(batch, dict)
    assert len(batch) == 2
    assert isinstance(batch[ct.VALUE], torch.Tensor)
    assert batch[ct.VALUE].shape == (batch_size, seq_len, 1)
    assert batch[ct.VALUE].dtype == torch.float
    assert isinstance(batch[ct.TIME], torch.Tensor)
    assert batch[ct.TIME].shape == (batch_size, seq_len, 1)
    assert batch[ct.TIME].dtype == torch.float


def test_timeseries_generate_same_random_seed() -> None:
    generator = TimeSeries(
        timeseries=timeseries.TimeSeries({"value": RandUniform(), "time": RandUniform()}),
        seq_len=Full(5),
    )
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


def test_timeseries_generate_different_random_seeds() -> None:
    generator = TimeSeries(
        timeseries=timeseries.TimeSeries({"value": RandUniform(), "time": RandUniform()}),
        seq_len=Full(5),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )
