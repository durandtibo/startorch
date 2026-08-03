from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch.example import BaseExampleGenerator, MoonsClassification, SwissRoll
from startorch.integration.iden.data.generator import ExampleDataGenerator
from startorch.testing import iden_available

SIZES = [1, 2, 4]


@pytest.fixture
def example() -> BaseExampleGenerator:
    return SwissRoll()


##########################################
#     Tests for ExampleDataGenerator     #
##########################################


@iden_available
def test_example_data_generator_repr(example: BaseExampleGenerator) -> None:
    assert repr(ExampleDataGenerator(example=example, batch_size=8, random_seed=1)).startswith(
        "ExampleDataGenerator("
    )


@iden_available
def test_example_data_generator_str(example: BaseExampleGenerator) -> None:
    assert str(ExampleDataGenerator(example=example, batch_size=8, random_seed=1)).startswith(
        "ExampleDataGenerator("
    )


@iden_available
def test_example_data_generator_equal_true(example: BaseExampleGenerator) -> None:
    assert ExampleDataGenerator(example=example, batch_size=8, random_seed=1).equal(
        ExampleDataGenerator(example=example, batch_size=8, random_seed=1)
    )


@iden_available
def test_example_data_generator_equal_false_different_example() -> None:
    assert not ExampleDataGenerator(example=SwissRoll(), batch_size=8, random_seed=1).equal(
        ExampleDataGenerator(example=MoonsClassification(), batch_size=8, random_seed=1)
    )


@iden_available
def test_example_data_generator_equal_false_different_batch_size(
    example: BaseExampleGenerator,
) -> None:
    assert not ExampleDataGenerator(example=example, batch_size=8, random_seed=1).equal(
        ExampleDataGenerator(example=example, batch_size=4, random_seed=1)
    )


@iden_available
def test_example_data_generator_equal_false_different_random_seed(
    example: BaseExampleGenerator,
) -> None:
    assert not ExampleDataGenerator(example=example, batch_size=8, random_seed=1).equal(
        ExampleDataGenerator(example=example, batch_size=8, random_seed=2)
    )


@iden_available
@pytest.mark.parametrize("batch_size", SIZES)
def test_example_data_generator_generate(example: BaseExampleGenerator, batch_size: int) -> None:
    generator = ExampleDataGenerator(example=example, batch_size=batch_size, random_seed=1)
    data = generator.generate()

    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (batch_size, 3)
    assert data[ct.FEATURE].dtype == torch.float


@iden_available
def test_example_data_generator_generate_multiple_calls(example: BaseExampleGenerator) -> None:
    generator = ExampleDataGenerator(example=example, batch_size=16, random_seed=1)
    assert not objects_are_equal(generator.generate(), generator.generate())


@iden_available
def test_example_data_generator_generate_same_random_seed() -> None:
    generator1 = ExampleDataGenerator(example=SwissRoll(), batch_size=16, random_seed=1)
    generator2 = ExampleDataGenerator(example=SwissRoll(), batch_size=16, random_seed=1)
    assert objects_are_equal(generator1.generate(), generator2.generate())


@iden_available
def test_example_data_generator_generate_different_random_seeds() -> None:
    generator1 = ExampleDataGenerator(example=SwissRoll(), batch_size=16, random_seed=1)
    generator2 = ExampleDataGenerator(example=SwissRoll(), batch_size=16, random_seed=2)
    assert not objects_are_equal(generator1.generate(), generator2.generate())
