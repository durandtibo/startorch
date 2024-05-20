from __future__ import annotations

import torch
from coola import objects_are_equal

from startorch.transformer.tensor import Identity

##############################
#     Tests for Identity     #
##############################


def test_identity_str() -> None:
    assert str(Identity()).startswith("IdentityTensorTransformer(")


def test_identity_transform() -> None:
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = Identity().transform(tensor)
    assert tensor is not out
    assert objects_are_equal(out, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_identity_transform_copy_false() -> None:
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = Identity(copy=False).transform(tensor)
    assert tensor is out
    assert objects_are_equal(out, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
