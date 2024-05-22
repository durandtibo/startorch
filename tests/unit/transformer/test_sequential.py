from __future__ import annotations

import torch
from coola import objects_are_equal

from startorch.transformer import Abs, Clamp, Sequential
from startorch.utils.seed import get_torch_generator

################################
#     Tests for Sequential     #
################################


def test_sequential_str() -> None:
    assert str(
        Sequential(
            [
                Abs(input="input", output="output1"),
                Clamp(input="input", output="output2", min=-1, max=2),
            ]
        )
    ).startswith("SequentialTransformer(")


def test_sequential_transform() -> None:
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    out = Sequential(
        [
            Abs(input="input", output="output1"),
            Clamp(input="input", output="output2", min=-1, max=2),
        ]
    ).transform(data)
    assert objects_are_equal(
        out,
        {
            "input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]),
            "output1": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "output2": torch.tensor([[1.0, -1.0, 2.0], [-1.0, 2.0, -1.0]]),
        },
    )


def test_sequential_transform_overwrite() -> None:
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    out = Sequential(
        [
            Clamp(input="input", output="input", min=-1, max=2, exist_ok=True),
            Abs(input="input", output="input", exist_ok=True),
        ]
    ).transform(data)
    assert objects_are_equal(out, {"input": torch.tensor([[1.0, 1.0, 2.0], [1.0, 2.0, 1.0]])})


def test_sequential_transform_same_random_seed() -> None:
    transformer = Sequential(
        [
            Abs(input="input", output="output1"),
            Clamp(input="input", output="output2", min=-1, max=2),
        ]
    )
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_sequential_transform_different_random_seeds() -> None:
    transformer = Sequential(
        [
            Abs(input="input", output="output1"),
            Clamp(input="input", output="output2", min=-1, max=2),
        ]
    )
    data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )
