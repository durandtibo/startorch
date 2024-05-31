from __future__ import annotations

import torch
from coola import objects_are_equal

from startorch.tensor.transformer import Abs, Clamp, Sequential
from startorch.utils.seed import get_torch_generator

################################
#     Tests for Sequential     #
################################


def test_sequential_str() -> None:
    assert str(Sequential([Abs(), Clamp(min=-1, max=2)])).startswith("SequentialTensorTransformer(")


def test_sequential_transform() -> None:
    out = Sequential([Clamp(min=-1, max=2), Abs()]).transform(
        torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    )
    assert objects_are_equal(out, torch.tensor([[1.0, 1.0, 2.0], [1.0, 2.0, 1.0]]))


def test_sequential_transform_same_random_seed() -> None:
    transformer = Sequential([Abs(), Clamp(min=-1, max=2)])
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_sequential_transform_different_random_seeds() -> None:
    transformer = Sequential([Abs(), Clamp(min=-1, max=2)])
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )
