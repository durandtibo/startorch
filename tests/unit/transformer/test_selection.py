from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.transformer import RemoveKeysTransformer, SelectKeysTransformer
from startorch.utils.seed import get_torch_generator

###########################################
#     Tests for RemoveKeysTransformer     #
###########################################


def test_remove_keys_transformer_str() -> None:
    assert str(RemoveKeysTransformer(keys=["input3"])).startswith("RemoveKeysTransformer(")


def test_remove_keys_transformer_transform_0_key() -> None:
    assert objects_are_equal(
        RemoveKeysTransformer(keys=[]).transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.ones(2, 4),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "input3": torch.ones(2, 4),
        },
    )


def test_remove_keys_transformer_transform_1_key() -> None:
    assert objects_are_equal(
        RemoveKeysTransformer(keys=["input3"]).transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.randn(2, 4),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
    )


def test_remove_keys_transformer_transform_2_keys() -> None:
    assert objects_are_equal(
        RemoveKeysTransformer(keys=["input3", "input2"]).transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.randn(2, 4),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
        },
    )


def test_remove_keys_transformer_transform_missing_ok_false() -> None:
    transformer = RemoveKeysTransformer(keys=["input3", "missing"])
    with pytest.raises(KeyError, match="missing is missing."):
        transformer.transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.randn(2, 4),
            }
        )


def test_remove_keys_transformer_transform_missing_ok_true() -> None:
    assert objects_are_equal(
        RemoveKeysTransformer(keys=["input3", "missing"], missing_ok=True).transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.randn(2, 4),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
    )


def test_remove_keys_transformer_transform_same_random_seed() -> None:
    transformer = RemoveKeysTransformer(keys=["input3"])
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12), "input3": torch.randn(2, 4)}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_remove_keys_transformer_transform_different_random_seeds() -> None:
    transformer = RemoveKeysTransformer(keys=["input3"])
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12), "input3": torch.randn(2, 4)}
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )


###########################################
#     Tests for SelectKeysTransformer     #
###########################################


def test_select_keys_transformer_str() -> None:
    assert str(SelectKeysTransformer(keys=["input1", "input2"])).startswith(
        "SelectKeysTransformer("
    )


def test_select_keys_transformer_transform_0_key() -> None:
    assert objects_are_equal(
        SelectKeysTransformer(keys=[]).transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.randn(2, 4),
            }
        ),
        {},
    )


def test_select_keys_transformer_transform_1_key() -> None:
    assert objects_are_equal(
        SelectKeysTransformer(keys=["input2"]).transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.randn(2, 4),
            }
        ),
        {"input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
    )


def test_select_keys_transformer_transform_2_keys() -> None:
    assert objects_are_equal(
        SelectKeysTransformer(keys=["input1", "input2"]).transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.randn(2, 4),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
    )


def test_select_keys_transformer_transform_missing_ok_false() -> None:
    transformer = SelectKeysTransformer(keys=["input1", "input2", "missing"])
    with pytest.raises(KeyError, match="missing is missing."):
        transformer.transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.randn(2, 4),
            }
        )


def test_select_keys_transformer_transform_missing_ok_true() -> None:
    assert objects_are_equal(
        SelectKeysTransformer(keys=["input1", "input2", "missing"], missing_ok=True).transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.randn(2, 4),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
    )


def test_select_keys_transformer_transform_same_random_seed() -> None:
    transformer = SelectKeysTransformer(keys=["input1", "input2"])
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12), "input3": torch.randn(2, 4)}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_select_keys_transformer_transform_different_random_seeds() -> None:
    transformer = SelectKeysTransformer(keys=["input1", "input2"])
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12), "input3": torch.randn(2, 4)}
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )
