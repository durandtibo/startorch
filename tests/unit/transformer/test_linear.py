from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.transformer import LinearTransformer, linear
from startorch.utils.seed import get_torch_generator

#######################################
#     Tests for LinearTransformer     #
#######################################


def test_linear_transformer_str() -> None:
    assert str(
        LinearTransformer(value="value", slope="slope", intercept="intercept", output="output")
    ).startswith("LinearTransformer(")


def test_linear_transformer_transform() -> None:
    assert objects_are_allclose(
        LinearTransformer(
            value="value", slope="slope", intercept="intercept", output="output"
        ).transform(
            {
                "value": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "slope": torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
                "intercept": torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
            }
        ),
        {
            "value": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "slope": torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
            "intercept": torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
            "output": torch.tensor([[3.0, 5.0, 7.0], [15.0, 19.0, 23.0]]),
        },
    )


def test_linear_transformer_transform_exist_ok_false() -> None:
    data = {
        "value": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "slope": torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
        "intercept": torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
        "output": 1,
    }
    transformer = LinearTransformer(
        value="value", slope="slope", intercept="intercept", output="output", exist_ok=False
    )
    with pytest.raises(KeyError, match="Key output already exists."):
        transformer.transform(data)


def test_linear_transformer_transform_exist_ok_true() -> None:
    data = {
        "value": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "slope": torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
        "intercept": torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
        "output": 1,
    }
    out = LinearTransformer(
        value="value", slope="slope", intercept="intercept", output="output", exist_ok=True
    ).transform(data)
    assert objects_are_equal(
        out,
        {
            "value": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "slope": torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
            "intercept": torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
            "output": torch.tensor([[3.0, 5.0, 7.0], [15.0, 19.0, 23.0]]),
        },
    )


def test_linear_transformer_transform_missing_key() -> None:
    transformer = LinearTransformer(
        value="value", slope="slope", intercept="intercept", output="output"
    )
    with pytest.raises(KeyError, match="Missing key: value."):
        transformer.transform({})


def test_linear_transformer_transform_same_random_seed() -> None:
    transformer = LinearTransformer(
        value="value", slope="slope", intercept="intercept", output="output"
    )
    data = {
        "value": torch.randn(4, 12),
        "slope": torch.randn(4, 12),
        "intercept": torch.randn(4, 12),
    }
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_linear_transformer_transform_different_random_seeds() -> None:
    transformer = LinearTransformer(
        value="value", slope="slope", intercept="intercept", output="output"
    )
    data = {
        "value": torch.randn(4, 12),
        "slope": torch.randn(4, 12),
        "intercept": torch.randn(4, 12),
    }
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )


############################
#     Tests for linear     #
############################


def test_linear() -> None:
    assert objects_are_allclose(
        linear(
            value=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            slope=torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
            intercept=torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
        ),
        torch.tensor([[3.0, 5.0, 7.0], [15.0, 19.0, 23.0]]),
    )
