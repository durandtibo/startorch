from unittest.mock import Mock

import torch
from coola import objects_are_equal
from pytest import mark
from redcat import BatchDict, BatchedTensorSeq

from startorch.sequence import BaseSequenceGenerator, RandUniform
from startorch.timeseries import MixedTimeSeries, TimeSeries
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2)


#####################################
#     Tests for MixedTimeSeries     #
#####################################


def test_mixed_timeseries_generator_str() -> None:
    assert str(
        MixedTimeSeries(
            TimeSeries({"key1": RandUniform(), "key2": RandUniform()}),
            key1="key1",
            key2="key2",
        )
    ).startswith("MixedTimeSeriesGenerator(")


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_mixed_timeseries_generator_generate(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = MixedTimeSeries(
        TimeSeries(
            {
                "key1": RandUniform(feature_size=feature_size),
                "key2": RandUniform(feature_size=feature_size),
            }
        ),
        key1="key1",
        key2="key2",
    ).generate(batch_size=batch_size, seq_len=seq_len)

    assert isinstance(batch, BatchDict)
    assert batch.batch_size == batch_size
    assert len(batch) == 2

    key1 = batch["key1"]
    assert isinstance(key1, BatchedTensorSeq)
    assert key1.batch_size == batch_size
    assert key1.seq_len == seq_len
    assert key1.data.shape == (batch_size, seq_len, feature_size)
    assert key1.data.dtype == torch.float

    key2 = batch["key2"]
    assert isinstance(key2, BatchedTensorSeq)
    assert key2.batch_size == batch_size
    assert key2.seq_len == seq_len
    assert key2.data.shape == (batch_size, seq_len, feature_size)
    assert key2.data.dtype == torch.float


def test_mixed_timeseries_generator_generate_mock() -> None:
    batch = MixedTimeSeries(
        TimeSeries(
            {
                "key1": Mock(
                    spec=BaseSequenceGenerator,
                    generate=Mock(
                        return_value=BatchedTensorSeq(
                            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
                        )
                    ),
                ),
                "key2": Mock(
                    spec=BaseSequenceGenerator,
                    generate=Mock(
                        return_value=BatchedTensorSeq(
                            torch.tensor([[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])
                        )
                    ),
                ),
            }
        ),
        key1="key1",
        key2="key2",
    ).generate(batch_size=2, seq_len=5)
    assert batch.equal(
        BatchDict(
            {
                "key1": BatchedTensorSeq(torch.tensor([[0, 11, 2, 13, 4], [5, 16, 7, 18, 9]])),
                "key2": BatchedTensorSeq(torch.tensor([[10, 1, 12, 3, 14], [15, 6, 17, 8, 19]])),
            }
        )
    )


def test_mixed_timeseries_generator_generate_same_random_seed() -> None:
    generator = MixedTimeSeries(
        TimeSeries({"key1": RandUniform(), "key2": RandUniform()}),
        key1="key1",
        key2="key2",
    )
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_mixed_timeseries_generator_generate_different_random_seeds() -> None:
    generator = MixedTimeSeries(
        TimeSeries({"key1": RandUniform(), "key2": RandUniform()}),
        key1="key1",
        key2="key2",
    )
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
