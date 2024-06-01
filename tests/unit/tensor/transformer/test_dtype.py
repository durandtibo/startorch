from __future__ import annotations

import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.tensor.transformer import FloatTensorTransformer, LongTensorTransformer
from startorch.utils.seed import get_torch_generator

############################################
#     Tests for FloatTensorTransformer     #
############################################


def test_float_str() -> None:
    assert str(FloatTensorTransformer()).startswith("FloatTensorTransformer(")


def test_float_transform() -> None:
    assert objects_are_allclose(
        FloatTensorTransformer().transform(torch.tensor([[1, -2, 3], [-4, 5, -6]])),
        torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
    )


def test_float_transform_same_random_seed() -> None:
    transformer = FloatTensorTransformer()
    tensor = torch.tensor([[1, -2, 3], [-4, 5, -6]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_float_transform_different_random_seeds() -> None:
    transformer = FloatTensorTransformer()
    tensor = torch.tensor([[1, -2, 3], [-4, 5, -6]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


###########################################
#     Tests for LongTensorTransformer     #
###########################################


def test_long_str() -> None:
    assert str(LongTensorTransformer()).startswith("LongTensorTransformer(")


def test_long_transform() -> None:
    assert objects_are_allclose(
        LongTensorTransformer().transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])),
        torch.tensor([[1, -2, 3], [-4, 5, -6]]),
    )


def test_long_transform_same_random_seed() -> None:
    transformer = LongTensorTransformer()
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_long_transform_different_random_seeds() -> None:
    transformer = LongTensorTransformer()
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )
