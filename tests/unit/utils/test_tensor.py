from __future__ import annotations

import pytest
import torch

from startorch.utils.tensor import shapes_are_equal

######################################
#     Tests for shapes_are_equal     #
######################################


def test_shapes_are_equal_0_tensor() -> None:
    assert not shapes_are_equal([])


def test_shapes_are_equal_1_tensor() -> None:
    assert shapes_are_equal([torch.rand(2, 3)])


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 4)])
def test_shapes_are_equal_true_2_tensors(shape: tuple[int, ...]) -> None:
    assert shapes_are_equal([torch.rand(*shape), torch.rand(*shape)])


def test_shapes_are_equal_true_3_tensors() -> None:
    assert shapes_are_equal([torch.rand(2, 3), torch.zeros(2, 3), torch.ones(2, 3)])


def test_shapes_are_equal_false_2_tensors() -> None:
    assert not shapes_are_equal([torch.rand(2, 3), torch.rand(2, 3, 1)])


def test_shapes_are_equal_false_3_tensors() -> None:
    assert not shapes_are_equal([torch.rand(2, 3), torch.zeros(2, 3, 4), torch.ones(2)])
