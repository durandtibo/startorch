from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.tensor.transformer import Abs
from startorch.transformer import TensorTransformer
from startorch.utils.seed import get_torch_generator

#######################################
#     Tests for TensorTransformer     #
#######################################


def test_tensor_transformer_str() -> None:
    assert str(TensorTransformer(transformer=Abs(), input="input", output="output")).startswith(
        "TensorTransformer("
    )


def test_tensor_transformer_transform() -> None:
    assert objects_are_allclose(
        TensorTransformer(transformer=Abs(), input="input", output="output").transform(
            {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
        ),
        {
            "input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
            "output": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
    )


def test_tensor_transformer_transform_exist_ok_false() -> None:
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]), "output": 1}
    transformer = TensorTransformer(
        transformer=Abs(), input="input", output="output", exist_ok=False
    )
    with pytest.raises(KeyError, match=r"Key output already exists."):
        transformer.transform(data)


def test_tensor_transformer_transform_exist_ok_true() -> None:
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]), "output": 1}
    out = TensorTransformer(
        transformer=Abs(), input="input", output="output", exist_ok=True
    ).transform(data)
    assert objects_are_equal(
        out,
        {
            "input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
            "output": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
    )


def test_tensor_transformer_transform_missing_key() -> None:
    transformer = TensorTransformer(transformer=Abs(), input="input", output="output")
    with pytest.raises(KeyError, match=r"Missing key: input."):
        transformer.transform({})


def test_tensor_transformer_transform_same_random_seed() -> None:
    transformer = TensorTransformer(transformer=Abs(), input="input", output="output")
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_tensor_transformer_transform_different_random_seeds() -> None:
    transformer = TensorTransformer(transformer=Abs(), input="input", output="output")
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )
