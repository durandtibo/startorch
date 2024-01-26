import pytest
import torch
from coola import objects_are_equal
from redcat import BatchDict, BatchedTensorSeq

from startorch import constants as ct
from startorch.sequence import RandNormal, RandUniform
from startorch.timeseries import Merge, TimeSeries
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


###########################
#     Tests for Merge     #
###########################


def test_merge_str() -> None:
    assert str(
        Merge(
            [
                TimeSeries({"value": RandUniform(), "time": RandUniform()}),
                TimeSeries({"value": RandNormal(), "time": RandNormal()}),
            ],
        )
    ).startswith("MergeTimeSeriesGenerator(")


def test_merge_generators() -> None:
    generator = Merge(
        [
            TimeSeries({"value": RandUniform(), "time": RandUniform()}),
            TimeSeries({"value": RandNormal(), "time": RandNormal()}),
        ],
    )
    assert len(generator._generators) == 2
    assert isinstance(generator._generators[0], TimeSeries)
    assert isinstance(generator._generators[1], TimeSeries)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_merge_generate(batch_size: int, seq_len: int) -> None:
    batch = Merge(
        [
            TimeSeries({"value": RandUniform(), "time": RandUniform()}),
            TimeSeries({"value": RandNormal(), "time": RandNormal()}),
        ],
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


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_merge_generate_time_0d(batch_size: int, seq_len: int) -> None:
    batch = Merge(
        [
            TimeSeries({"value": RandUniform(), "time": RandUniform(feature_size=[])}),
            TimeSeries({"value": RandNormal(), "time": RandNormal(feature_size=[])}),
        ],
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
    assert batch_time.data.shape == (batch_size, seq_len)
    assert batch_time.data.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_merge_generate_value_0d(batch_size: int, seq_len: int) -> None:
    batch = Merge(
        [
            TimeSeries({"value": RandUniform(feature_size=[]), "time": RandUniform()}),
            TimeSeries({"value": RandNormal(feature_size=[]), "time": RandNormal()}),
        ],
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchDict)
    assert batch.batch_size == batch_size
    assert len(batch.data) == 2

    batch_value = batch.data[ct.VALUE]
    assert isinstance(batch_value, BatchedTensorSeq)
    assert batch_value.batch_size == batch_size
    assert batch_value.seq_len == seq_len
    assert batch_value.data.shape == (batch_size, seq_len)
    assert batch_value.data.dtype == torch.float

    batch_time = batch.data[ct.TIME]
    assert isinstance(batch_time, BatchedTensorSeq)
    assert batch_time.batch_size == batch_size
    assert batch_time.seq_len == seq_len
    assert batch_time.data.shape == (batch_size, seq_len, 1)
    assert batch_time.data.dtype == torch.float


def test_merge_generate_same_random_seed() -> None:
    generator = Merge(
        [
            TimeSeries({"value": RandUniform(), "time": RandUniform()}),
            TimeSeries({"value": RandNormal(), "time": RandNormal()}),
        ],
    )
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_merge_generate_different_random_seeds() -> None:
    generator = Merge(
        [
            TimeSeries({"value": RandUniform(), "time": RandUniform()}),
            TimeSeries({"value": RandNormal(), "time": RandNormal()}),
        ],
    )
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
