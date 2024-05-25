from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch.tensor import Full, RandUniform
from startorch.timeseries import TensorTimeSeriesGenerator
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


###############################################
#     Tests for TensorTimeSeriesGenerator     #
###############################################


def test_tensor_timeseries_generator_str() -> None:
    assert str(
        TensorTimeSeriesGenerator(
            generators={"value": RandUniform(), "time": RandUniform()},
        )
    ).startswith("TensorTimeSeriesGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_tensor_timeseries_generator_generate(batch_size: int, seq_len: int) -> None:
    batch = TensorTimeSeriesGenerator(
        generators={"value": RandUniform(), "time": RandUniform()},
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, dict)
    assert len(batch) == 2
    assert isinstance(batch[ct.VALUE], torch.Tensor)
    assert batch[ct.VALUE].shape == (batch_size, seq_len)
    assert batch[ct.VALUE].dtype == torch.float
    assert isinstance(batch[ct.TIME], torch.Tensor)
    assert batch[ct.TIME].shape == (batch_size, seq_len)
    assert batch[ct.TIME].dtype == torch.float


def test_tensor_timeseries_generator_generate_size_empty() -> None:
    batch = TensorTimeSeriesGenerator(
        generators={"value": Full(1), "time": Full(2.0)},
    ).generate(seq_len=10, batch_size=5)
    assert objects_are_equal(
        batch,
        {
            "value": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            "time": torch.tensor(
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                ]
            ),
        },
    )


def test_tensor_timeseries_generator_generate_size_3() -> None:
    batch = TensorTimeSeriesGenerator(
        generators={"value": Full(1), "time": Full(2.0)},
        size=(3,),
    ).generate(seq_len=3, batch_size=5)
    assert objects_are_equal(
        batch,
        {
            "value": torch.tensor(
                [
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                ]
            ),
            "time": torch.tensor(
                [
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                ]
            ),
        },
    )


def test_tensor_timeseries_generator_generate_empty() -> None:
    assert objects_are_equal(
        TensorTimeSeriesGenerator(generators={}).generate(seq_len=10, batch_size=5),
        {},
    )


def test_tensor_timeseries_generator_generate_same_random_seed() -> None:
    generator = TensorTimeSeriesGenerator(
        generators={"value": RandUniform(), "time": RandUniform()},
        size=(10,),
    )
    assert objects_are_equal(
        generator.generate(seq_len=32, batch_size=64, rng=get_torch_generator(1)),
        generator.generate(seq_len=32, batch_size=64, rng=get_torch_generator(1)),
    )


def test_tensor_timeseries_generator_generate_different_random_seeds() -> None:
    generator = TensorTimeSeriesGenerator(
        generators={"value": RandUniform(), "time": RandUniform()},
        size=(10,),
    )
    assert not objects_are_equal(
        generator.generate(seq_len=32, batch_size=64, rng=get_torch_generator(1)),
        generator.generate(seq_len=32, batch_size=64, rng=get_torch_generator(2)),
    )
