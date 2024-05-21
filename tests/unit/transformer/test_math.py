from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.transformer import Abs, Clamp
from startorch.utils.seed import get_torch_generator

#########################
#     Tests for Abs     #
#########################


def test_abs_str() -> None:
    assert str(Abs(input="input", output="output")).startswith("AbsTransformer(")


def test_abs_transform() -> None:
    assert objects_are_allclose(
        Abs(input="input", output="output").transform(
            {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
        ),
        {
            "input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
            "output": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
    )


def test_abs_transform_exist_ok_false() -> None:
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]), "output": 1}
    transformer = Abs(input="input", output="output", exist_ok=False)
    with pytest.raises(KeyError, match="Key output already exists."):
        transformer.transform(data)


def test_abs_transform_exist_ok_true() -> None:
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]), "output": 1}
    out = Abs(input="input", output="output", exist_ok=True).transform(data)
    assert objects_are_equal(
        out,
        {
            "input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
            "output": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
    )


def test_abs_transform_missing_key() -> None:
    transformer = Abs(input="input", output="output")
    with pytest.raises(KeyError, match="Missing key: input."):
        transformer.transform({})


def test_abs_transform_same_random_seed() -> None:
    transformer = Abs(input="input", output="output")
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_abs_transform_different_random_seeds() -> None:
    transformer = Abs(input="input", output="output")
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )


###########################
#     Tests for Clamp     #
###########################


def test_clamp_str() -> None:
    assert str(Clamp(input="input", output="output", min=-2, max=2)).startswith("ClampTransformer(")


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
def test_clamp_min(min_value: float) -> None:
    assert Clamp(input="input", output="output", min=min_value, max=None)._min == min_value


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_clamp_max(max_value: float) -> None:
    assert Clamp(input="input", output="output", min=None, max=max_value)._max == max_value


def test_clamp_incorrect_min_max() -> None:
    with pytest.raises(ValueError, match="`min` and `max` cannot be both None"):
        Clamp(input="input", output="output", min=None, max=None)


def test_clamp_transform() -> None:
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    out = Clamp(
        input="input",
        output="output",
        min=-2,
        max=2,
    ).transform(data)
    assert objects_are_equal(
        out,
        {
            "input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
            "output": torch.tensor([[1.0, -2.0, 2.0], [-2.0, 2.0, -2.0]]),
        },
    )


def test_clamp_transform_only_min_value() -> None:
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    out = Clamp(input="input", output="output", min=-1, max=None).transform(data)
    assert objects_are_equal(
        out,
        {
            "input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
            "output": torch.tensor([[1.0, -1.0, 3.0], [-1.0, 5.0, -1.0]]),
        },
    )


def test_clamp_transform_only_max_value() -> None:
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    out = Clamp(input="input", output="output", min=None, max=-1).transform(data)
    assert objects_are_equal(
        out,
        {
            "input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
            "output": torch.tensor([[-1.0, -2.0, -1.0], [-4.0, -1.0, -6.0]]),
        },
    )


def test_clamp_transform_same_random_seed() -> None:
    transformer = Clamp(input="input", output="output", min=-2, max=2)
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_clamp_transform_different_random_seeds() -> None:
    transformer = Clamp(input="input", output="output", min=-2, max=2)
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )
