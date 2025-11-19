from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch
from coola import objects_are_equal

from startorch.utils.mask import mask_by_row, mask_square_matrix
from startorch.utils.seed import get_torch_generator

#################################
#     Tests for mask_by_row     #
#################################


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_mask_by_row_long(n: int) -> None:
    tensor = mask_by_row(tensor=torch.arange(20).view(4, 5).add(1), n=n)
    assert tensor.dtype == torch.long
    assert tensor.shape == (4, 5)
    assert objects_are_equal(tensor.eq(0).sum(dim=1), torch.tensor([n, n, n, n]))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_mask_by_row_float(n: int) -> None:
    tensor = mask_by_row(tensor=torch.arange(20, dtype=torch.float).view(4, 5).add(1.0), n=n)
    assert tensor.dtype == torch.float
    assert tensor.shape == (4, 5)
    assert objects_are_equal(tensor.eq(0).sum(dim=1), torch.tensor([n, n, n, n]))


def test_mask_by_row_mock() -> None:
    mock = Mock(return_value=torch.tensor([3, 0, 1, 4, 2]))
    with patch("startorch.utils.mask.torch.randperm", mock):
        tensor = mask_by_row(tensor=torch.arange(20).view(4, 5).add(1), n=2)
        mock.assert_called_with(5, generator=None)
        assert objects_are_equal(
            tensor,
            torch.tensor(
                [
                    [0, 2, 3, 0, 5],
                    [0, 7, 8, 0, 10],
                    [0, 12, 13, 0, 15],
                    [0, 17, 18, 0, 20],
                ]
            ),
        )


def test_mask_by_row_mask_value() -> None:
    mock = Mock(return_value=torch.tensor([3, 0, 1, 4, 2]))
    with patch("startorch.utils.mask.torch.randperm", mock):
        tensor = mask_by_row(tensor=torch.arange(20).view(4, 5).add(1), n=2, mask_value=-1)
        assert objects_are_equal(
            tensor,
            torch.tensor(
                [
                    [-1, 2, 3, -1, 5],
                    [-1, 7, 8, -1, 10],
                    [-1, 12, 13, -1, 15],
                    [-1, 17, 18, -1, 20],
                ]
            ),
        )


def test_mask_by_row_float_incorrect_shape_1d() -> None:
    with pytest.raises(
        ValueError, match=r"Expected a 2D tensor but received a tensor with 1 dimensions"
    ):
        mask_by_row(tensor=torch.ones(5), n=2)


def test_mask_by_row_float_incorrect_shape_3d() -> None:
    with pytest.raises(
        ValueError, match=r"Expected a 2D tensor but received a tensor with 3 dimensions"
    ):
        mask_by_row(tensor=torch.ones(2, 5, 2), n=2)


def test_mask_by_row_float_incorrect_n_low() -> None:
    with pytest.raises(ValueError, match=r"Incorrect number of values to mask: -1"):
        mask_by_row(tensor=torch.ones(2, 5), n=-1)


def test_mask_by_row_float_incorrect_n_high() -> None:
    with pytest.raises(ValueError, match=r"Incorrect number of values to mask: 6"):
        mask_by_row(tensor=torch.ones(2, 5), n=6)


def test_mask_by_row_same_random_seed() -> None:
    tensor = torch.arange(20).view(4, 5).add(1)
    assert objects_are_equal(
        mask_by_row(tensor, n=2, rng=get_torch_generator(1)),
        mask_by_row(tensor, n=2, rng=get_torch_generator(1)),
    )


def test_mask_by_row_different_random_seeds() -> None:
    tensor = torch.arange(20).view(4, 5).add(1)
    assert not objects_are_equal(
        mask_by_row(tensor, n=2, rng=get_torch_generator(1)),
        mask_by_row(tensor, n=2, rng=get_torch_generator(2)),
    )


########################################
#     Tests for mask_square_matrix     #
########################################


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_mask_square_matrix_long(n: int) -> None:
    matrix = mask_square_matrix(matrix=torch.arange(25).view(5, 5).add(1), n=n)
    assert matrix.dtype == torch.long
    assert matrix.shape == (5, 5)
    assert objects_are_equal(matrix.eq(0).sum(dim=0), torch.full((5,), fill_value=n))
    assert objects_are_equal(matrix.eq(0).sum(dim=1), torch.full((5,), fill_value=n))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_mask_square_matrix_float(n: int) -> None:
    matrix = mask_square_matrix(matrix=torch.arange(25, dtype=torch.float).view(5, 5).add(1), n=n)
    assert matrix.dtype == torch.float
    assert matrix.shape == (5, 5)
    assert objects_are_equal(matrix.eq(0).sum(dim=0), torch.full((5,), fill_value=n))
    assert objects_are_equal(matrix.eq(0).sum(dim=1), torch.full((5,), fill_value=n))


def test_mask_square_matrix_mask_value() -> None:
    mock = Mock(return_value=torch.tensor([3, 0, 1, 4, 2]))
    with patch("startorch.utils.mask.torch.randperm", mock):
        tensor = mask_square_matrix(matrix=torch.arange(25).view(5, 5).add(1), n=2, mask_value=-1)
        assert objects_are_equal(
            tensor,
            torch.tensor(
                [
                    [-1, 2, 3, -1, 5],
                    [6, -1, -1, 9, 10],
                    [11, 12, -1, 14, -1],
                    [16, -1, 18, -1, 20],
                    [-1, 22, 23, 24, -1],
                ]
            ),
        )


def test_mask_square_matrix_incorrect_dims() -> None:
    with pytest.raises(
        ValueError, match=r"Expected a 2D tensor but received a tensor with 1 dimensions"
    ):
        mask_square_matrix(torch.ones(5), n=1)


def test_mask_square_matrix_incorrect_shape() -> None:
    with pytest.raises(ValueError, match=r"Expected a square matrix but received"):
        mask_square_matrix(torch.ones(5, 4), n=1)


@pytest.mark.parametrize("n", [-1, 6])
def test_mask_square_matrix_incorrect_n(n: int) -> None:
    with pytest.raises(ValueError, match=r"Incorrect number of values to mask:"):
        mask_square_matrix(torch.ones(5, 5), n=n)


def test_mask_square_matrix_same_random_seed() -> None:
    matrix = torch.arange(25).view(5, 5).add(1)
    assert objects_are_equal(
        mask_square_matrix(matrix, n=2, rng=get_torch_generator(1)),
        mask_square_matrix(matrix, n=2, rng=get_torch_generator(1)),
    )


def test_mask_square_matrix_different_random_seeds() -> None:
    matrix = torch.arange(25).view(5, 5).add(1)
    assert not objects_are_equal(
        mask_square_matrix(matrix, n=2, rng=get_torch_generator(1)),
        mask_square_matrix(matrix, n=2, rng=get_torch_generator(2)),
    )
