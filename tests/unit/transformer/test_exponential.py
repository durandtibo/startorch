from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.transformer import Exponential
from startorch.utils.seed import get_torch_generator

#################################
#     Tests for Exponential     #
#################################


def test_exponential_str() -> None:
    assert str(Exponential(rate="rate", output="output")).startswith("ExponentialTransformer(")


def test_exponential_transform() -> None:
    data = {"rate": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    out = Exponential(rate="rate", output="output").transform(data)
    assert data is not out
    assert isinstance(out, dict)
    assert len(out) == 2
    assert objects_are_equal(out["rate"], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert out["output"].shape == (2, 3)
    assert out["output"].dtype == torch.float


def test_exponential_transform_exist_ok_false() -> None:
    data = {"rate": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), "output": 1}
    transformer = Exponential(rate="rate", output="output", exist_ok=False)
    with pytest.raises(KeyError, match="Key output already exists."):
        transformer.transform(data)


def test_exponential_transform_exist_ok_true() -> None:
    data = {"rate": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), "output": 1}
    out = Exponential(rate="rate", output="output", exist_ok=True).transform(data)
    assert data is not out
    assert isinstance(out, dict)
    assert len(out) == 2
    assert objects_are_equal(out["rate"], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert out["output"].shape == (2, 3)
    assert out["output"].dtype == torch.float


def test_exponential_transform_missing_key() -> None:
    transformer = Exponential(rate="rate", output="output")
    with pytest.raises(KeyError, match="Missing key: rate."):
        transformer.transform({})


def test_exponential_transform_same_random_seed() -> None:
    transformer = Exponential(rate="rate", output="output")
    data = {"rate": torch.ones(2, 10)}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_exponential_transform_different_random_seeds() -> None:
    transformer = Exponential(rate="rate", output="output")
    data = {"rate": torch.ones(2, 10)}
    assert not objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )