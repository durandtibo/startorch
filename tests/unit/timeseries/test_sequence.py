import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch.sequence import RandUniform, UniformCategorical
from startorch.timeseries import SequenceTimeSeries
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


########################################
#     Tests for SequenceTimeSeries     #
########################################


def test_timeseries_generator_str() -> None:
    assert str(SequenceTimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})).startswith(
        "SequenceTimeSeriesGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_timeseries_generator_generate_float(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = SequenceTimeSeries(
        {ct.VALUE: RandUniform(feature_size=feature_size), ct.TIME: RandUniform()},
    ).generate(batch_size=batch_size, seq_len=seq_len)

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


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_timeseries_generator_generate_long(batch_size: int, seq_len: int) -> None:
    batch = SequenceTimeSeries(
        {ct.VALUE: UniformCategorical(num_categories=10), ct.TIME: RandUniform()}
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(batch, dict)
    assert len(batch) == 2

    batch_value = batch[ct.VALUE]
    assert isinstance(batch_value, torch.Tensor)
    assert batch_value.shape == (batch_size, seq_len)
    assert batch_value.dtype == torch.long

    batch_time = batch[ct.TIME]
    assert isinstance(batch_time, torch.Tensor)
    assert batch_time.shape == (batch_size, seq_len, 1)
    assert batch_time.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_timeseries_generator_generate_1(batch_size: int, seq_len: int) -> None:
    batch = SequenceTimeSeries(
        {ct.VALUE: RandUniform()},
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(batch, dict)
    assert len(batch) == 1

    batch_value = batch[ct.VALUE]
    assert isinstance(batch_value, torch.Tensor)
    assert batch_value.shape == (batch_size, seq_len, 1)
    assert batch_value.dtype == torch.float


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_timeseries_generator_generate_3(batch_size: int, seq_len: int) -> None:
    batch = SequenceTimeSeries(
        {
            ct.VALUE: RandUniform(),
            ct.TIME: RandUniform(),
            "3": RandUniform(),
        },
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(batch, dict)
    assert len(batch) == 3

    batch_value = batch[ct.VALUE]
    assert isinstance(batch_value, torch.Tensor)
    assert batch_value.shape == (batch_size, seq_len, 1)
    assert batch_value.dtype == torch.float

    batch_time = batch[ct.TIME]
    assert isinstance(batch_time, torch.Tensor)
    assert batch_time.shape == (batch_size, seq_len, 1)
    assert batch_time.dtype == torch.float

    batch_3 = batch["3"]
    assert isinstance(batch_3, torch.Tensor)
    assert batch_3.shape == (batch_size, seq_len, 1)
    assert batch_3.dtype == torch.float


def test_timeseries_generator_generate_same_random_seed() -> None:
    generator = SequenceTimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_timeseries_generator_generate_different_random_seeds() -> None:
    generator = SequenceTimeSeries({ct.VALUE: RandUniform(), ct.TIME: RandUniform()})
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
