from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.sequence import Full, RandInt, RandUniform
from startorch.timeseries import Concatenate, SequenceTimeSeriesGenerator
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


####################################################
#     Tests for ConcatenateTimeSeriesGenerator     #
####################################################


def test_concatenate_str() -> None:
    assert str(
        Concatenate(
            generators=[
                SequenceTimeSeriesGenerator(
                    generators={"value": RandUniform(), "time": RandUniform()},
                ),
                SequenceTimeSeriesGenerator(generators={"label": RandInt(0, 10)}),
            ]
        )
    ).startswith("ConcatenateTimeSeriesGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_concatenate_generate(batch_size: int, seq_len: int) -> None:
    batch = Concatenate(
        generators=[
            SequenceTimeSeriesGenerator(
                generators={"value": RandUniform(), "time": RandUniform()},
            ),
            SequenceTimeSeriesGenerator(generators={"label": RandInt(0, 10)}),
        ]
    ).generate(seq_len=seq_len, batch_size=batch_size)
    assert isinstance(batch, dict)
    assert len(batch) == 3
    assert isinstance(batch["value"], torch.Tensor)
    assert batch["value"].shape == (batch_size, seq_len, 1)
    assert batch["value"].dtype == torch.float
    assert isinstance(batch["time"], torch.Tensor)
    assert batch["time"].shape == (batch_size, seq_len, 1)
    assert batch["time"].dtype == torch.float
    assert isinstance(batch["label"], torch.Tensor)
    assert batch["label"].shape == (batch_size, seq_len)
    assert batch["label"].dtype == torch.long


def test_concatenate_generate_no_randomness() -> None:
    batch = Concatenate(
        generators=[
            SequenceTimeSeriesGenerator(
                generators={"value": Full(1, feature_size=()), "time": Full(2.0, feature_size=())},
            ),
            SequenceTimeSeriesGenerator(generators={"label": Full(42, feature_size=())}),
        ]
    ).generate(batch_size=5, seq_len=3)
    assert objects_are_equal(
        batch,
        {
            "value": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            "time": torch.tensor(
                [
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                ]
            ),
            "label": torch.tensor(
                [
                    [42, 42, 42],
                    [42, 42, 42],
                    [42, 42, 42],
                    [42, 42, 42],
                    [42, 42, 42],
                ]
            ),
        },
    )


def test_concatenate_generate_empty() -> None:
    assert Concatenate(generators=[]).generate(seq_len=32, batch_size=5) == {}


def test_concatenate_generate_same_random_seed() -> None:
    generator = Concatenate(
        generators=[
            SequenceTimeSeriesGenerator(
                generators={"value": RandUniform(), "time": RandUniform()},
            ),
            SequenceTimeSeriesGenerator(generators={"label": RandInt(0, 10)}),
        ]
    )
    assert objects_are_equal(
        generator.generate(seq_len=32, batch_size=64, rng=get_torch_generator(1)),
        generator.generate(seq_len=32, batch_size=64, rng=get_torch_generator(1)),
    )


def test_concatenate_generate_different_random_seeds() -> None:
    generator = Concatenate(
        generators=[
            SequenceTimeSeriesGenerator(
                generators={"value": RandUniform(), "time": RandUniform()},
            ),
            SequenceTimeSeriesGenerator(generators={"label": RandInt(0, 10)}),
        ]
    )
    assert not objects_are_equal(
        generator.generate(seq_len=32, batch_size=64, rng=get_torch_generator(1)),
        generator.generate(seq_len=32, batch_size=64, rng=get_torch_generator(2)),
    )
