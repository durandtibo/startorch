from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from coola import objects_are_equal

from startorch.utils.conversion import to_array, to_tensor, to_tuple

if TYPE_CHECKING:
    from collections.abc import Sequence

##############################
#     Tests for to_array     #
##############################


@pytest.mark.parametrize(
    "data", [np.array([1, 2, 4, 0]), torch.tensor([1, 2, 4, 0]), [1, 2, 4, 0], (1, 2, 4, 0)]
)
def test_to_array(data: Sequence | torch.Tensor | np.ndarray) -> None:
    assert objects_are_equal(to_array(data), np.array([1, 2, 4, 0]))


###############################
#     Tests for to_tensor     #
###############################


@pytest.mark.parametrize(
    "data", [np.array([1, 2, 4, 0]), torch.tensor([1, 2, 4, 0]), [1, 2, 4, 0], (1, 2, 4, 0)]
)
def test_to_tensor(data: Sequence | torch.Tensor | np.ndarray) -> None:
    assert objects_are_equal(to_tensor(data), torch.tensor([1, 2, 4, 0]))


##############################
#     Tests for to_tuple     #
##############################


def test_to_tuples_tuple() -> None:
    assert to_tuple((1, 2, 3)) == (1, 2, 3)


def test_to_tuples_list() -> None:
    assert to_tuple([1, 2, 3]) == (1, 2, 3)


def test_to_tuples_bool() -> None:
    assert to_tuple(True) == (True,)


def test_to_tuples_int() -> None:
    assert to_tuple(1) == (1,)


def test_to_tuples_float() -> None:
    assert to_tuple(42.1) == (42.1,)


def test_to_tuples_str() -> None:
    assert to_tuple("abc") == ("abc",)
