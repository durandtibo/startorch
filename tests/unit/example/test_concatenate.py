from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.example import ConcatenateExampleGenerator, TensorExampleGenerator
from startorch.tensor import Full, RandInt, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


#################################################
#     Tests for ConcatenateExampleGenerator     #
#################################################


def test_concatenate_str() -> None:
    assert str(
        ConcatenateExampleGenerator(
            generators=[
                TensorExampleGenerator(
                    generators={"value": RandUniform(), "time": RandUniform()},
                    size=(6,),
                ),
                TensorExampleGenerator(
                    generators={"label": RandInt(0, 10)},
                    size=(),
                ),
            ]
        )
    ).startswith("ConcatenateExampleGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_concatenate_generate(batch_size: int, feature_size: int) -> None:
    batch = ConcatenateExampleGenerator(
        generators=[
            TensorExampleGenerator(
                generators={"value": RandUniform(), "time": RandUniform()},
                size=(feature_size,),
            ),
            TensorExampleGenerator(
                generators={"label": RandInt(0, 10)},
                size=(),
            ),
        ]
    ).generate(batch_size=batch_size)
    assert isinstance(batch, dict)
    assert len(batch) == 3
    assert isinstance(batch["value"], torch.Tensor)
    assert batch["value"].shape == (batch_size, feature_size)
    assert batch["value"].dtype == torch.float
    assert isinstance(batch["time"], torch.Tensor)
    assert batch["time"].shape == (batch_size, feature_size)
    assert batch["time"].dtype == torch.float
    assert isinstance(batch["label"], torch.Tensor)
    assert batch["label"].shape == (batch_size,)
    assert batch["label"].dtype == torch.long


def test_concatenate_generate_no_randomness() -> None:
    batch = ConcatenateExampleGenerator(
        generators=[
            TensorExampleGenerator(
                generators={"value": Full(1), "time": Full(2.0)},
                size=(3,),
            ),
            TensorExampleGenerator(
                generators={"label": Full(42)},
                size=(),
            ),
        ]
    ).generate(batch_size=5)
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
            "label": torch.tensor([42, 42, 42, 42, 42]),
        },
    )


def test_concatenate_generate_empty() -> None:
    assert ConcatenateExampleGenerator(generators=[]).generate(batch_size=5) == {}


def test_concatenate_generate_same_random_seed() -> None:
    generator = ConcatenateExampleGenerator(
        generators=[
            TensorExampleGenerator(
                generators={"value": RandUniform(), "time": RandUniform()},
                size=(6,),
            ),
            TensorExampleGenerator(
                generators={"label": RandInt(0, 10)},
                size=(),
            ),
        ]
    )
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


def test_concatenate_generate_different_random_seeds() -> None:
    generator = ConcatenateExampleGenerator(
        generators=[
            TensorExampleGenerator(
                generators={"value": RandUniform(), "time": RandUniform()},
                size=(6,),
            ),
            TensorExampleGenerator(
                generators={"label": RandInt(0, 10)},
                size=(),
            ),
        ]
    )
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )
