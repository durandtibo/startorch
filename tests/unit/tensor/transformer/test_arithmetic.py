from __future__ import annotations

import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.tensor.transformer import AddTensorTransformer, MulTensorTransformer
from startorch.utils.seed import get_torch_generator

##########################################
#     Tests for AddTensorTransformer     #
##########################################


def test_add_str() -> None:
    assert str(AddTensorTransformer(value=1)).startswith("AddTensorTransformer(")


def test_add_transform() -> None:
    assert objects_are_allclose(
        AddTensorTransformer(value=1).transform(
            torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
        ),
        torch.tensor([[2.0, -1.0, 4.0], [-3.0, 6.0, -5.0]]),
    )


def test_add_transform_same_random_seed() -> None:
    transformer = AddTensorTransformer(value=1)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_add_transform_different_random_seeds() -> None:
    transformer = AddTensorTransformer(value=1)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


##########################################
#     Tests for MulTensorTransformer     #
##########################################


def test_mul_str() -> None:
    assert str(MulTensorTransformer(value=2)).startswith("MulTensorTransformer(")


def test_mul_transform() -> None:
    assert objects_are_allclose(
        MulTensorTransformer(value=2).transform(
            torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
        ),
        torch.tensor([[2.0, -4.0, 6.0], [-8.0, 10.0, -12.0]]),
    )


def test_mul_transform_same_random_seed() -> None:
    transformer = MulTensorTransformer(value=2)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_mul_transform_different_random_seeds() -> None:
    transformer = MulTensorTransformer(value=2)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )
