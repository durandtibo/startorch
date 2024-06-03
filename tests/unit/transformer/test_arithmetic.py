from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.transformer import AddTransformer, MulTransformer, SubTransformer
from startorch.utils.seed import get_torch_generator

####################################
#     Tests for AddTransformer     #
####################################


def test_add_transformer_str() -> None:
    assert str(AddTransformer(inputs=["input1", "input2"], output="output")).startswith(
        "AddTransformer("
    )


def test_add_transformer_inputs_empty() -> None:
    with pytest.raises(ValueError, match="inputs cannot be empty"):
        AddTransformer(inputs=[], output="output")


def test_add_transformer_transform_1_input() -> None:
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


def test_add_transformer_transform_2_inputs() -> None:
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


def test_add_transformer_transform_3_inputs() -> None:
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


def test_add_transformer_transform_exist_ok_false() -> None:
    data = {
        "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
        "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "output": 1,
    }
    transformer = AddTransformer(inputs=["input1", "input2"], output="output", exist_ok=False)
    with pytest.raises(KeyError, match="Key output already exists."):
        transformer.transform(data)


def test_add_transformer_transform_exist_ok_true() -> None:
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


def test_add_transformer_transform_missing_key() -> None:
    transformer = AddTransformer(inputs=["input1", "input2"], output="output")
    with pytest.raises(KeyError, match="Missing key: input1."):
        transformer.transform({})


def test_add_transformer_transform_same_random_seed() -> None:
    transformer = AddTransformer(inputs=["input1", "input2"], output="output")
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12)}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_add_transformer_transform_different_random_seeds() -> None:
    transformer = AddTransformer(inputs=["input1", "input2"], output="output")
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12)}
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )


####################################
#     Tests for MulTransformer     #
####################################


def test_mul_transformer_str() -> None:
    assert str(MulTransformer(inputs=["input1", "input2"], output="output")).startswith(
        "MulTransformer("
    )


def test_mul_transformer_inputs_empty() -> None:
    with pytest.raises(ValueError, match="inputs cannot be empty"):
        MulTransformer(inputs=[], output="output")


def test_mul_transformer_transform_1_input() -> None:
    assert objects_are_allclose(
        MulTransformer(inputs=["input1"], output="output").transform(
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


def test_mul_transformer_transform_2_inputs() -> None:
    assert objects_are_allclose(
        MulTransformer(inputs=["input1", "input2"], output="output").transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "output": torch.tensor([[0.0, -2.0, 6.0], [-16.0, 25.0, -36.0]]),
        },
    )


def test_mul_transformer_transform_3_inputs() -> None:
    assert objects_are_allclose(
        MulTransformer(inputs=["input1", "input2", "input3"], output="output").transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "input3": torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "input3": torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            "output": torch.tensor([[0.0, -4.0, 12.0], [-32.0, 50.0, -72.0]]),
        },
    )


def test_mul_transformer_transform_exist_ok_false() -> None:
    data = {
        "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
        "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "output": 1,
    }
    transformer = MulTransformer(inputs=["input1", "input2"], output="output", exist_ok=False)
    with pytest.raises(KeyError, match="Key output already exists."):
        transformer.transform(data)


def test_mul_transformer_transform_exist_ok_true() -> None:
    data = {
        "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
        "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "output": 1,
    }
    out = MulTransformer(inputs=["input1", "input2"], output="output", exist_ok=True).transform(
        data
    )
    assert objects_are_equal(
        out,
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "output": torch.tensor([[0.0, -2.0, 6.0], [-16.0, 25.0, -36.0]]),
        },
    )


def test_mul_transformer_transform_missing_key() -> None:
    transformer = MulTransformer(inputs=["input1", "input2"], output="output")
    with pytest.raises(KeyError, match="Missing key: input1."):
        transformer.transform({})


def test_mul_transformer_transform_same_random_seed() -> None:
    transformer = MulTransformer(inputs=["input1", "input2"], output="output")
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12)}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_mul_transformer_transform_different_random_seeds() -> None:
    transformer = MulTransformer(inputs=["input1", "input2"], output="output")
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12)}
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )


####################################
#     Tests for SubTransformer     #
####################################


def test_sub_transformer_str() -> None:
    assert str(SubTransformer(minuend="input1", subtrahend="input2", output="output")).startswith(
        "SubTransformer("
    )


def test_sub_transformer_transform() -> None:
    assert objects_are_allclose(
        SubTransformer(minuend="input1", subtrahend="input2", output="output").transform(
            {
                "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
                "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            }
        ),
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "output": torch.tensor([[-1.0, -3.0, -1.0], [-8.0, 0.0, -12.0]]),
        },
    )


def test_sub_transformer_transform_exist_ok_false() -> None:
    data = {
        "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
        "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "output": 1,
    }
    transformer = SubTransformer(
        minuend="input1", subtrahend="input2", output="output", exist_ok=False
    )
    with pytest.raises(KeyError, match="Key output already exists."):
        transformer.transform(data)


def test_sub_transformer_transform_exist_ok_true() -> None:
    data = {
        "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
        "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "output": 1,
    }
    out = SubTransformer(
        minuend="input1", subtrahend="input2", output="output", exist_ok=True
    ).transform(data)
    assert objects_are_equal(
        out,
        {
            "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
            "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "output": torch.tensor([[-1.0, -3.0, -1.0], [-8.0, 0.0, -12.0]]),
        },
    )


def test_sub_transformer_transform_missing_key() -> None:
    transformer = SubTransformer(minuend="input1", subtrahend="input2", output="output")
    with pytest.raises(KeyError, match="Missing key: input1."):
        transformer.transform({})


def test_sub_transformer_transform_same_random_seed() -> None:
    transformer = SubTransformer(minuend="input1", subtrahend="input2", output="output")
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12)}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_sub_transformer_transform_different_random_seeds() -> None:
    transformer = SubTransformer(minuend="input1", subtrahend="input2", output="output")
    data = {"input1": torch.randn(4, 12), "input2": torch.randn(4, 12)}
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )
