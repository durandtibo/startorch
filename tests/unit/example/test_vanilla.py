from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.example import VanillaExampleGenerator

SIZES = (1, 2, 4)


#############################################
#     Tests for VanillaExampleGenerator     #
#############################################


def test_vanilla_str() -> None:
    assert str(
        VanillaExampleGenerator({"value": torch.ones(10, 3), "time": torch.arange(10)})
    ).startswith("VanillaExampleGenerator(")


def test_vanilla_incorrect_data() -> None:
    with pytest.raises(ValueError, match=r"data cannot be empty"):
        VanillaExampleGenerator({})


@pytest.mark.parametrize("batch_size", SIZES)
def test_vanilla_batch_size(batch_size: int) -> None:
    assert (
        VanillaExampleGenerator(
            {"value": torch.ones(batch_size, 3), "time": torch.arange(batch_size)}
        )._batch_size
        == batch_size
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4, 10])
def test_vanilla_lower_batch_size(batch_size: int) -> None:
    out = VanillaExampleGenerator({"value": torch.ones(10, 3), "time": torch.arange(10)}).generate(
        batch_size=batch_size
    )
    assert objects_are_equal(
        out, {"value": torch.ones(batch_size, 3), "time": torch.arange(batch_size)}
    )


def test_vanilla_larger_batch_size() -> None:
    generator = VanillaExampleGenerator({"value": torch.ones(10, 3), "time": torch.arange(10)})
    with pytest.raises(RuntimeError, match=r"Incorrect batch_size: 11."):
        generator.generate(batch_size=11)
