from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.tensor.transformer import AbsTensorTransformer, ClampTensorTransformer
from startorch.utils.seed import get_torch_generator

##########################################
#     Tests for AbsTensorTransformer     #
##########################################


def test_abs_str() -> None:
    assert str(AbsTensorTransformer()).startswith("AbsTensorTransformer(")


def test_abs_transform() -> None:
    assert objects_are_allclose(
        AbsTensorTransformer().transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])),
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )


def test_abs_transform_same_random_seed() -> None:
    transformer = AbsTensorTransformer()
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_abs_transform_different_random_seeds() -> None:
    transformer = AbsTensorTransformer()
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


############################################
#     Tests for ClampTensorTransformer     #
############################################


def test_clamp_str() -> None:
    assert str(ClampTensorTransformer(min=-2, max=2)).startswith("ClampTensorTransformer(")


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
def test_clamp_min(min_value: float) -> None:
    assert ClampTensorTransformer(min=min_value, max=None)._min == min_value


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_clamp_max(max_value: float) -> None:
    assert ClampTensorTransformer(min=None, max=max_value)._max == max_value


def test_clamp_incorrect_min_max() -> None:
    with pytest.raises(ValueError, match="`min` and `max` cannot be both None"):
        ClampTensorTransformer(min=None, max=None)


def test_clamp_transform() -> None:
    out = ClampTensorTransformer(
        min=-2,
        max=2,
    ).transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    assert objects_are_equal(out, torch.tensor([[1.0, -2.0, 2.0], [-2.0, 2.0, -2.0]]))


def test_clamp_transform_only_min_value() -> None:
    out = ClampTensorTransformer(min=-1, max=None).transform(
        torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    )
    assert objects_are_equal(out, torch.tensor([[1.0, -1.0, 3.0], [-1.0, 5.0, -1.0]]))


def test_clamp_transform_only_max_value() -> None:
    out = ClampTensorTransformer(min=None, max=-1).transform(
        torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    )
    assert objects_are_equal(out, torch.tensor([[-1.0, -2.0, -1.0], [-4.0, -1.0, -6.0]]))


def test_clamp_transform_same_random_seed() -> None:
    transformer = ClampTensorTransformer(min=-2, max=2)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_clamp_transform_different_random_seeds() -> None:
    transformer = ClampTensorTransformer(min=-2, max=2)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )
