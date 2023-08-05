import torch
from coola import objects_are_equal
from pytest import mark
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


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_timeseries_generator_generate_float(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    timeseries = TimeSeries(
        {ct.VALUE: RandUniform(feature_size=feature_size), ct.TIME: RandUniform()},
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(timeseries, BatchDict)
    assert timeseries.batch_size == batch_size
    assert len(timeseries.data) == 2

    assert isinstance(timeseries.data[ct.VALUE], BatchedTensorSeq)
    assert timeseries.data[ct.VALUE].batch_size == batch_size
    assert timeseries.data[ct.VALUE].seq_len == seq_len
    assert timeseries.data[ct.VALUE].data.shape == (batch_size, seq_len, feature_size)
    assert timeseries.data[ct.VALUE].data.dtype == torch.float

    assert isinstance(timeseries.data[ct.TIME], BatchedTensorSeq)
    assert timeseries.data[ct.TIME].batch_size == batch_size
    assert timeseries.data[ct.TIME].seq_len == seq_len
    assert timeseries.data[ct.TIME].data.shape == (batch_size, seq_len, 1)
    assert timeseries.data[ct.TIME].data.dtype == torch.float


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_timeseries_generator_generate_long(batch_size: int, seq_len: int) -> None:
    timeseries = TimeSeries(
        {ct.VALUE: UniformCategorical(num_categories=10), ct.TIME: RandUniform()}
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(timeseries, BatchDict)
    assert timeseries.batch_size == batch_size
    assert len(timeseries.data) == 2

    assert isinstance(timeseries.data[ct.VALUE], BatchedTensorSeq)
    assert timeseries.data[ct.VALUE].batch_size == batch_size
    assert timeseries.data[ct.VALUE].seq_len == seq_len
    assert timeseries.data[ct.VALUE].data.shape == (batch_size, seq_len)
    assert timeseries.data[ct.VALUE].data.dtype == torch.long

    assert isinstance(timeseries.data[ct.TIME], BatchedTensorSeq)
    assert timeseries.data[ct.TIME].batch_size == batch_size
    assert timeseries.data[ct.TIME].seq_len == seq_len
    assert timeseries.data[ct.TIME].data.shape == (batch_size, seq_len, 1)
    assert timeseries.data[ct.TIME].data.dtype == torch.float


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_timeseries_generator_generate_1(batch_size: int, seq_len: int) -> None:
    timeseries = TimeSeries(
        {ct.VALUE: RandUniform()},
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(timeseries, BatchDict)
    assert timeseries.batch_size == batch_size
    assert len(timeseries.data) == 1

    assert isinstance(timeseries.data[ct.VALUE], BatchedTensorSeq)
    assert timeseries.data[ct.VALUE].batch_size == batch_size
    assert timeseries.data[ct.VALUE].seq_len == seq_len
    assert timeseries.data[ct.VALUE].data.shape == (batch_size, seq_len, 1)
    assert timeseries.data[ct.VALUE].data.dtype == torch.float


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_timeseries_generator_generate_3(batch_size: int, seq_len: int) -> None:
    timeseries = TimeSeries(
        {
            ct.VALUE: RandUniform(),
            ct.TIME: RandUniform(),
            "3": RandUniform(),
        },
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(timeseries, BatchDict)
    assert timeseries.batch_size == batch_size
    assert len(timeseries.data) == 3

    assert isinstance(timeseries.data[ct.VALUE], BatchedTensorSeq)
    assert timeseries.data[ct.VALUE].batch_size == batch_size
    assert timeseries.data[ct.VALUE].seq_len == seq_len
    assert timeseries.data[ct.VALUE].data.shape == (batch_size, seq_len, 1)
    assert timeseries.data[ct.VALUE].data.dtype == torch.float

    assert isinstance(timeseries.data[ct.TIME], BatchedTensorSeq)
    assert timeseries.data[ct.TIME].batch_size == batch_size
    assert timeseries.data[ct.TIME].seq_len == seq_len
    assert timeseries.data[ct.TIME].data.shape == (batch_size, seq_len, 1)
    assert timeseries.data[ct.TIME].data.dtype == torch.float

    assert isinstance(timeseries.data["3"], BatchedTensorSeq)
    assert timeseries.data["3"].batch_size == batch_size
    assert timeseries.data["3"].seq_len == seq_len
    assert timeseries.data["3"].data.shape == (batch_size, seq_len, 1)
    assert timeseries.data["3"].data.dtype == torch.float


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
