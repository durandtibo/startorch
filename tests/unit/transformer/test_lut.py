from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.transformer import LookupTable
from startorch.utils.seed import get_torch_generator

#################################
#     Tests for LookupTable     #
#################################


def test_lookup_table_str() -> None:
    assert str(
        LookupTable(
            weights=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), index="index", output="output"
        )
    ).startswith("LookupTableTransformer(")


def test_lookup_table_transform() -> None:
    data = {"index": torch.tensor([1, 2, 3])}
    out = LookupTable(
        weights=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), index="index", output="output"
    ).transform(data)
    assert objects_are_equal(
        out,
        {"index": torch.tensor([1, 2, 3]), "output": torch.tensor([4.0, 3.0, 2.0])},
    )


def test_lookup_table_transform_weights_1d() -> None:
    data = {"index": torch.tensor([[1, 2, 3], [4, 0, 2]])}
    out = LookupTable(
        weights=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), index="index", output="output"
    ).transform(data)
    assert objects_are_equal(
        out,
        {
            "index": torch.tensor([[1, 2, 3], [4, 0, 2]]),
            "output": torch.tensor([[4.0, 3.0, 2.0], [1.0, 5.0, 3.0]]),
        },
    )


def test_lookup_table_transform_weights_2d() -> None:
    data = {"index": torch.tensor([[1, 2, 3], [4, 0, 2]])}
    out = LookupTable(
        weights=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
            ]
        ),
        index="index",
        output="output",
    ).transform(data)
    assert objects_are_equal(
        out,
        {
            "index": torch.tensor([[1, 2, 3], [4, 0, 2]]),
            "output": torch.tensor(
                [
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                    [[4.0, 4.0, 4.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
                ]
            ),
        },
    )


def test_lookup_table_transform_exist_ok_false() -> None:
    data = {"index": torch.tensor([[1, 2, 3], [4, 0, 2]]), "output": 1}
    transformer = LookupTable(
        weights=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
        index="index",
        output="output",
        exist_ok=False,
    )
    with pytest.raises(KeyError, match="Key output already exists."):
        transformer.transform(data)


def test_lookup_table_transform_exist_ok_true() -> None:
    data = {"index": torch.tensor([[1, 2, 3], [4, 0, 2]]), "output": 1}
    out = LookupTable(
        weights=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
        index="index",
        output="output",
        exist_ok=True,
    ).transform(data)
    assert objects_are_equal(
        out,
        {
            "index": torch.tensor([[1, 2, 3], [4, 0, 2]]),
            "output": torch.tensor([[4.0, 3.0, 2.0], [1.0, 5.0, 3.0]]),
        },
    )


def test_lookup_table_transform_missing_key() -> None:
    transformer = LookupTable(
        weights=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), index="index", output="output"
    )
    with pytest.raises(KeyError, match="Missing key: index."):
        transformer.transform({})


def test_lookup_table_transform_same_random_seed() -> None:
    transformer = LookupTable(
        weights=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), index="index", output="output"
    )
    data = {"index": torch.randint(0, 6, (2, 10))}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_lookup_table_transform_different_random_seeds() -> None:
    transformer = LookupTable(
        weights=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), index="index", output="output"
    )
    data = {"index": torch.randint(0, 6, (2, 10))}
    # the outputs are equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )
