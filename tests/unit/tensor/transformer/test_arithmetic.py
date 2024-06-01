from __future__ import annotations

import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.tensor.transformer import (
    AddTensorTransformer,
    DivTensorTransformer,
    FmodTensorTransformer,
    MulTensorTransformer,
)
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
#     Tests for DivTensorTransformer     #
##########################################


def test_div_str() -> None:
    assert str(DivTensorTransformer(divisor=4)).startswith("DivTensorTransformer(")


def test_div_transform() -> None:
    assert objects_are_allclose(
        DivTensorTransformer(divisor=4).transform(
            torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
        ),
        torch.tensor([[0.25, -0.5, 0.75], [-1.0, 1.25, -1.5]]),
    )


def test_div_transform_trunc() -> None:
    assert objects_are_allclose(
        DivTensorTransformer(divisor=4, rounding_mode="trunc").transform(
            torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
        ),
        torch.tensor([[0.0, -0.0, 0.0], [-1.0, 1.0, -1.0]]),
    )


def test_div_transform_floor() -> None:
    assert objects_are_allclose(
        DivTensorTransformer(divisor=4, rounding_mode="floor").transform(
            torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
        ),
        torch.tensor([[0.0, -1.0, 0.0], [-1.0, 1.0, -2.0]]),
    )


def test_div_transform_same_random_seed() -> None:
    transformer = DivTensorTransformer(divisor=4)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_div_transform_different_random_seeds() -> None:
    transformer = DivTensorTransformer(divisor=4)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


###########################################
#     Tests for FmodTensorTransformer     #
###########################################


def test_fmod_str() -> None:
    assert str(FmodTensorTransformer(divisor=4)).startswith("FmodTensorTransformer(")


def test_fmod_transform() -> None:
    assert objects_are_allclose(
        FmodTensorTransformer(divisor=4).transform(
            torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
        ),
        torch.tensor([[1.0, -2.0, 3.0], [0.0, 1.0, -2.0]]),
    )


def test_fmod_transform_same_random_seed() -> None:
    transformer = FmodTensorTransformer(divisor=4)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_fmod_transform_different_random_seeds() -> None:
    transformer = FmodTensorTransformer(divisor=4)
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
