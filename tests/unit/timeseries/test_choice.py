import pytest
import torch
from coola import objects_are_equal
from redcat import BatchDict, BatchedTensorSeq

from startorch import constants as ct
from startorch.sequence import RandNormal, RandUniform
from startorch.timeseries import MultinomialChoice, TimeSeries
from startorch.utils.seed import get_torch_generator
from startorch.utils.weight import GENERATOR, WEIGHT

SIZES = (1, 2)


#######################################
#     Tests for MultinomialChoice     #
#######################################


def test_multinomial_choice_str() -> None:
    assert str(
        MultinomialChoice(
            (
                {
                    WEIGHT: 2.0,
                    GENERATOR: TimeSeries({"value": RandUniform(), "time": RandUniform()}),
                },
                {
                    WEIGHT: 1.0,
                    GENERATOR: TimeSeries({"value": RandNormal(), "time": RandNormal()}),
                },
            ),
        )
    ).startswith("MultinomialChoiceTimeSeriesGenerator(")


def test_multinomial_choice_generators() -> None:
    generator = MultinomialChoice(
        (
            {
                WEIGHT: 2.0,
                GENERATOR: TimeSeries({"value": RandUniform(), "time": RandUniform()}),
            },
            {
                WEIGHT: 1.0,
                GENERATOR: TimeSeries({"value": RandNormal(), "time": RandNormal()}),
            },
        ),
    )
    assert len(generator._generators) == 2
    assert isinstance(generator._generators[0], TimeSeries)
    assert isinstance(generator._generators[1], TimeSeries)


def test_multinomial_choice_weights() -> None:
    assert MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: TimeSeries({"value": RandUniform(), "time": RandUniform()})},
            {WEIGHT: 1.0, GENERATOR: TimeSeries({"value": RandUniform(), "time": RandUniform()})},
            {WEIGHT: 3.0, GENERATOR: TimeSeries({"value": RandUniform(), "time": RandUniform()})},
        )
    )._weights.equal(torch.tensor([2.0, 1.0, 3.0]))


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_multinomial_choice_generate(batch_size: int, seq_len: int) -> None:
    batch = MultinomialChoice(
        (
            {
                WEIGHT: 2.0,
                GENERATOR: TimeSeries({"value": RandUniform(), "time": RandUniform()}),
            },
            {
                WEIGHT: 1.0,
                GENERATOR: TimeSeries({"value": RandNormal(), "time": RandNormal()}),
            },
        ),
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


def test_multinomial_choice_generate_same_random_seed() -> None:
    generator = MultinomialChoice(
        (
            {
                WEIGHT: 2.0,
                GENERATOR: TimeSeries({"value": RandUniform(), "time": RandUniform()}),
            },
            {
                WEIGHT: 1.0,
                GENERATOR: TimeSeries({"value": RandNormal(), "time": RandNormal()}),
            },
        ),
    )
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_multinomial_choice_generate_different_random_seeds() -> None:
    generator = MultinomialChoice(
        (
            {
                WEIGHT: 2.0,
                GENERATOR: TimeSeries({"value": RandUniform(), "time": RandUniform()}),
            },
            {
                WEIGHT: 1.0,
                GENERATOR: TimeSeries({"value": RandNormal(), "time": RandNormal()}),
            },
        ),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
