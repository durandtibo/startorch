import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch.sequence import RandNormal, RandUniform
from startorch.timeseries import Merge, SequenceTimeSeries
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


###########################
#     Tests for Merge     #
###########################


def test_merge_str() -> None:
    assert str(
        Merge(
            [
                SequenceTimeSeries({"value": RandUniform(), "time": RandUniform()}),
                SequenceTimeSeries({"value": RandNormal(), "time": RandNormal()}),
            ],
        )
    ).startswith("MergeTimeSeriesGenerator(")


def test_merge_generators() -> None:
    generator = Merge(
        [
            SequenceTimeSeries({"value": RandUniform(), "time": RandUniform()}),
            SequenceTimeSeries({"value": RandNormal(), "time": RandNormal()}),
        ],
    )
    assert len(generator._generators) == 2
    assert isinstance(generator._generators[0], SequenceTimeSeries)
    assert isinstance(generator._generators[1], SequenceTimeSeries)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_merge_generate(batch_size: int, seq_len: int) -> None:
    batch = Merge(
        [
            SequenceTimeSeries({"value": RandUniform(), "time": RandUniform()}),
            SequenceTimeSeries({"value": RandNormal(), "time": RandNormal()}),
        ],
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


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_merge_generate_time_0d(batch_size: int, seq_len: int) -> None:
    batch = Merge(
        [
            SequenceTimeSeries({"value": RandUniform(), "time": RandUniform(feature_size=[])}),
            SequenceTimeSeries({"value": RandNormal(), "time": RandNormal(feature_size=[])}),
        ],
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, dict)
    assert len(batch) == 2

    batch_value = batch[ct.VALUE]
    assert isinstance(batch_value, torch.Tensor)
    assert batch_value.shape == (batch_size, seq_len, 1)
    assert batch_value.dtype == torch.float

    batch_time = batch[ct.TIME]
    assert isinstance(batch_time, torch.Tensor)
    assert batch_time.shape == (batch_size, seq_len)
    assert batch_time.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_merge_generate_value_0d(batch_size: int, seq_len: int) -> None:
    batch = Merge(
        [
            SequenceTimeSeries({"value": RandUniform(feature_size=[]), "time": RandUniform()}),
            SequenceTimeSeries({"value": RandNormal(feature_size=[]), "time": RandNormal()}),
        ],
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, dict)
    assert len(batch) == 2

    batch_value = batch[ct.VALUE]
    assert isinstance(batch_value, torch.Tensor)
    assert batch_value.shape == (batch_size, seq_len)
    assert batch_value.dtype == torch.float

    batch_time = batch[ct.TIME]
    assert isinstance(batch_time, torch.Tensor)
    assert batch_time.shape == (batch_size, seq_len, 1)
    assert batch_time.dtype == torch.float


def test_merge_generate_same_random_seed() -> None:
    generator = Merge(
        [
            SequenceTimeSeries({"value": RandUniform(), "time": RandUniform()}),
            SequenceTimeSeries({"value": RandNormal(), "time": RandNormal()}),
        ],
    )
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_merge_generate_different_random_seeds() -> None:
    generator = Merge(
        [
            SequenceTimeSeries({"value": RandUniform(), "time": RandUniform()}),
            SequenceTimeSeries({"value": RandNormal(), "time": RandNormal()}),
        ],
    )
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
