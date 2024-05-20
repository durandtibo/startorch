from __future__ import annotations

import torch
from coola import objects_are_equal

from startorch.transformer import Identity
from startorch.utils.seed import get_torch_generator

##############################
#     Tests for Identity     #
##############################


def test_identity_str() -> None:
    assert str(Identity()).startswith("IdentityTransformer(")


def test_identity_transform() -> None:
    data = {"key": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    out = Identity().transform(data)
    assert data is not out
    assert objects_are_equal(out, data)


def test_identity_transform_copy_false() -> None:
    data = {"key": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    out = Identity(copy=False).transform(data)
    assert data is out
    assert objects_are_equal(out, data)


def test_identity_transform_same_random_seed() -> None:
    transformer = Identity()
    data = {"key": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_identity_transform_different_random_seeds() -> None:
    transformer = Identity()
    data = {"key": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )
