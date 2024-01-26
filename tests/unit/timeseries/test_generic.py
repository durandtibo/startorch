import pytest
import torch
from coola import objects_are_equal
from redcat import BatchDict, BatchedTensorSeq

from startorch import constants as ct
from startorch.sequence import RandUniform, UniformCategorical
from startorch.timeseries import TimeSeries
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2)


################################
#     Tests for TimeSeries     #
################################


def test_timeseries_generator_str() -> None:
    assert str(TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})).startswith(
        "TimeSeriesGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_timeseries_generator_generate_float(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = TimeSeries(
        {ct.VALUE: RandUniform(feature_size=feature_size), ct.TIME: RandUniform()},
    ).generate(batch_size=batch_size, seq_len=seq_len)

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


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_timeseries_generator_generate_long(batch_size: int, seq_len: int) -> None:
    batch = TimeSeries(
        {ct.VALUE: UniformCategorical(num_categories=10), ct.TIME: RandUniform()}
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(batch, BatchDict)
    assert batch.batch_size == batch_size
    assert len(batch.data) == 2

    batch_value = batch.data[ct.VALUE]
    assert isinstance(batch_value, BatchedTensorSeq)
    assert batch_value.batch_size == batch_size
    assert batch_value.seq_len == seq_len
    assert batch_value.data.shape == (batch_size, seq_len)
    assert batch_value.data.dtype == torch.long

    batch_time = batch.data[ct.TIME]
    assert isinstance(batch_time, BatchedTensorSeq)
    assert batch_time.batch_size == batch_size
    assert batch_time.seq_len == seq_len
    assert batch_time.data.shape == (batch_size, seq_len, 1)
    assert batch_time.data.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_timeseries_generator_generate_1(batch_size: int, seq_len: int) -> None:
    batch = TimeSeries(
        {ct.VALUE: RandUniform()},
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(batch, BatchDict)
    assert batch.batch_size == batch_size
    assert len(batch.data) == 1

    batch_value = batch.data[ct.VALUE]
    assert isinstance(batch_value, BatchedTensorSeq)
    assert batch_value.batch_size == batch_size
    assert batch_value.seq_len == seq_len
    assert batch_value.data.shape == (batch_size, seq_len, 1)
    assert batch_value.data.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_timeseries_generator_generate_3(batch_size: int, seq_len: int) -> None:
    batch = TimeSeries(
        {
            ct.VALUE: RandUniform(),
            ct.TIME: RandUniform(),
            "3": RandUniform(),
        },
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(batch, BatchDict)
    assert batch.batch_size == batch_size
    assert len(batch.data) == 3

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

    batch_3 = batch.data["3"]
    assert isinstance(batch_3, BatchedTensorSeq)
    assert batch_3.batch_size == batch_size
    assert batch_3.seq_len == seq_len
    assert batch_3.data.shape == (batch_size, seq_len, 1)
    assert batch_3.data.dtype == torch.float


def test_timeseries_generator_generate_same_random_seed() -> None:
    generator = TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_timeseries_generator_generate_different_random_seeds() -> None:
    generator = TimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
