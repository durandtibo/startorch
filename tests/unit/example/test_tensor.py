from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch.example import TensorExampleGenerator
from startorch.tensor import Full, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


############################################
#     Tests for TensorExampleGenerator     #
############################################


def test_tensor_example_generator_str() -> None:
    assert str(
        TensorExampleGenerator(
            generators={"value": RandUniform(), "time": RandUniform()},
            size=(10,),
        )
    ).startswith("TensorExampleGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_tensor_example_generator_generate(batch_size: int, feature_size: int) -> None:
    batch = TensorExampleGenerator(
        generators={"value": RandUniform(), "time": RandUniform()},
        size=(feature_size,),
    ).generate(batch_size=batch_size)
    assert isinstance(batch, dict)
    assert len(batch) == 2
    assert isinstance(batch[ct.VALUE], torch.Tensor)
    assert batch[ct.VALUE].shape == (batch_size, feature_size)
    assert batch[ct.VALUE].dtype == torch.float
    assert isinstance(batch[ct.TIME], torch.Tensor)
    assert batch[ct.TIME].shape == (batch_size, feature_size)
    assert batch[ct.TIME].dtype == torch.float


def test_tensor_example_generator_generate_size_empty() -> None:
    batch = TensorExampleGenerator(
        generators={"value": Full(1), "time": Full(2.0)},
    ).generate(batch_size=5)
    assert objects_are_equal(
        batch,
        {
            "value": torch.tensor([1, 1, 1, 1, 1]),
            "time": torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0]),
        },
    )


def test_tensor_example_generator_generate_size_3() -> None:
    batch = TensorExampleGenerator(
        generators={"value": Full(1), "time": Full(2.0)},
        size=(3,),
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
        },
    )


def test_tensor_example_generator_generate_empty() -> None:
    assert objects_are_equal(
        TensorExampleGenerator(generators={}).generate(batch_size=5),
        {},
    )


def test_tensor_example_generator_generate_same_random_seed() -> None:
    generator = TensorExampleGenerator(
        generators={"value": RandUniform(), "time": RandUniform()},
        size=(10,),
    )
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


def test_tensor_example_generator_generate_different_random_seeds() -> None:
    generator = TensorExampleGenerator(
        generators={"value": RandUniform(), "time": RandUniform()},
        size=(10,),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )
