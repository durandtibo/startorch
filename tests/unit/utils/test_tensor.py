from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.utils.tensor import circulant, shapes_are_equal

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


###############################
#     Tests for circulant     #
###############################


def test_circulant_dim_1() -> None:
    assert objects_are_equal(circulant(torch.tensor([1])), torch.tensor([[1]]))


def test_circulant_dim_2() -> None:
    assert objects_are_equal(circulant(torch.tensor([1, 2])), torch.tensor([[1, 2], [2, 1]]))


def test_circulant_dim_4() -> None:
    assert objects_are_equal(
        circulant(torch.tensor([1, 2, 3, 0])),
        torch.tensor([[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]]),
    )


def test_circulant_long() -> None:
    assert objects_are_equal(
        circulant(torch.tensor([1, 2, 3, 0])),
        torch.tensor([[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]]),
    )


def test_circulant_float() -> None:
    assert objects_are_equal(
        circulant(torch.tensor([1.0, 2.0, 3.0, 0.0])),
        torch.tensor(
            [[1.0, 2.0, 3.0, 0.0], [0.0, 1.0, 2.0, 3.0], [3.0, 0.0, 1.0, 2.0], [2.0, 3.0, 0.0, 1.0]]
        ),
    )


def test_circulant_incorrect_shape() -> None:
    with pytest.raises(ValueError, match=r"Expected a vector but received a 2-d tensor"):
        circulant(torch.tensor([[1, 2]]))
