from __future__ import annotations

import pytest
import torch
from batchtensor.nested import slice_along_seq
from coola import objects_are_equal

from startorch import constants as ct
from startorch.periodic.timeseries import Repeat
from startorch.sequence import RandUniform
from startorch.timeseries import TimeSeries
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]
DTYPES = (torch.float, torch.long)


############################
#     Tests for Repeat     #
############################


def test_repeat_str() -> None:
    assert str(Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}))).startswith(
        "RepeatPeriodicTimeSeriesGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("period", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_repeat_generate(batch_size: int, seq_len: int, period: int, feature_size: int) -> None:
    batch = Repeat(
        TimeSeries({ct.VALUE: RandUniform(feature_size=feature_size), ct.TIME: RandUniform()})
    ).generate(batch_size=batch_size, seq_len=seq_len, period=period)

    assert isinstance(batch, dict)
    assert len(batch) == 2

    batch_value = batch[ct.VALUE]
    assert isinstance(batch_value, torch.Tensor)
    assert batch_value.shape == (batch_size, seq_len, feature_size)
    assert batch_value.dtype == torch.float

    batch_time = batch[ct.TIME]
    assert isinstance(batch_time, torch.Tensor)
    assert batch_time.shape == (batch_size, seq_len, 1)
    assert batch_time.dtype == torch.float


def test_repeat_period_3() -> None:
    batch = Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})).generate(
        batch_size=2, period=3, seq_len=10
    )
    assert isinstance(batch, dict)
    assert objects_are_equal(
        slice_along_seq(batch, start=0, stop=3), slice_along_seq(batch, start=3, stop=6)
    )
    assert objects_are_equal(
        slice_along_seq(batch, start=0, stop=3), slice_along_seq(batch, start=6, stop=9)
    )


def test_repeat_period_4() -> None:
    batch = Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})).generate(
        batch_size=2, period=4, seq_len=10
    )
    assert isinstance(batch, dict)
    assert objects_are_equal(
        slice_along_seq(batch, start=0, stop=4), slice_along_seq(batch, start=4, stop=8)
    )
    assert objects_are_equal(
        slice_along_seq(batch, start=0, stop=2), slice_along_seq(batch, start=8)
    )


def test_repeat_generate_same_random_seed() -> None:
    generator = Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}))
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)),
    )


def test_repeat_generate_different_random_seeds() -> None:
    generator = Repeat(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()}))
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(2)),
    )
