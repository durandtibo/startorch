from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.transformer import AddTransformer
from startorch.utils.seed import get_torch_generator

####################################
#     Tests for AddTransformer     #
####################################


def test_tensor_transformer_str() -> None:
    assert str(AddTransformer(inputs=["input1", "input2"], output="output")).startswith(
        "AddTransformer("
    )


def test_tensor_transformer_inputs_empty() -> None:
    with pytest.raises(ValueError, match="inputs cannot be empty"):
        AddTransformer(inputs=[], output="output")


def test_tensor_transformer_transform_1_input() -> None:
    assert objects_are_allclose(
        AddTransformer(inputs=["input1"], output="output").transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "output": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
        },
    )


def test_tensor_transformer_transform_2_inputs() -> None:
    assert objects_are_allclose(
        AddTransformer(inputs=["input1", "input2"], output="output").transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "output": torch.tensor([[1.0, 1.0, 5.0], [0.0, 10.0, 0.0]]),
        },
    )


def test_tensor_transformer_transform_3_inputs() -> None:
    assert objects_are_allclose(
        AddTransformer(inputs=["input1", "input2", "input3"], output="output").transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "input3": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            "output": torch.tensor([[2.0, 2.0, 6.0], [1.0, 11.0, 1.0]]),
        },
    )


def test_tensor_transformer_transform_exist_ok_false() -> None:
    data = {
        "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
        "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "output": 1,
    }
    transformer = AddTransformer(inputs=["input1", "input2"], output="output", exist_ok=False)
    with pytest.raises(KeyError, match="Key output already exists."):
        transformer.transform(data)


def test_tensor_transformer_transform_exist_ok_true() -> None:
    data = {
        "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
        "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "output": 1,
    }
    out = AddTransformer(inputs=["input1", "input2"], output="output", exist_ok=True).transform(
        data
    )
    assert objects_are_equal(
        out,
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "output": torch.tensor([[1.0, 1.0, 5.0], [0.0, 10.0, 0.0]]),
        },
    )


def test_tensor_transformer_transform_missing_key() -> None:
    transformer = AddTransformer(inputs=["input1", "input2"], output="output")
    with pytest.raises(KeyError, match="Missing key: input1."):
        transformer.transform({})


def test_tensor_transformer_transform_same_random_seed() -> None:
    transformer = AddTransformer(inputs=["input1", "input2"], output="output")
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12)}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_tensor_transformer_transform_different_random_seeds() -> None:
    transformer = AddTransformer(inputs=["input1", "input2"], output="output")
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12)}
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )
