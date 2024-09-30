r"""Contain utility functions for PyTorch tensors."""

from __future__ import annotations

__all__ = ["circulant", "shapes_are_equal"]

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


def shapes_are_equal(tensors: Sequence[torch.Tensor]) -> bool:
    r"""Return ``True`` if the shapes of several tensors are equal,
    otherwise ``False``.

    This method does not check the values or the data type of the
    tensors.

    Args:
        tensors: The tensors to check.

    Returns:
        ``True`` if all the tensors have the same shape, otherwise
            ``False``. By design, this function returns ``False`` if
            no tensor is provided.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.utils.tensor import shapes_are_equal
    >>> shapes_are_equal([torch.rand(2, 3), torch.rand(2, 3)])
    True
    >>> shapes_are_equal([torch.rand(2, 3), torch.rand(2, 3, 1)])
    False

    ```
    """
    if not tensors:
        return False
    shape = tensors[0].shape
    return all(shape == tensor.shape for tensor in tensors[1:])


def circulant(vector: torch.Tensor) -> torch.Tensor:
    r"""Return a circulant matrix of shape ``(n, n)``.

    Args:
        vector: The base vector of shape ``(n,)`` used to generate
            the circulant matrix.

    Returns:
        A circulant matrix of shape ``(n, n)``.

    Raises:
        ValueError: if the input tensor is not a vector.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.utils.tensor import circulant
    >>> c = circulant(torch.arange(5))
    >>> c
    tensor([[0, 1, 2, 3, 4],
            [4, 0, 1, 2, 3],
            [3, 4, 0, 1, 2],
            [2, 3, 4, 0, 1],
            [1, 2, 3, 4, 0]])
    >>> c = circulant(torch.tensor([1, 2, 3, 0]))
    >>> c
    tensor([[1, 2, 3, 0],
            [0, 1, 2, 3],
            [3, 0, 1, 2],
            [2, 3, 0, 1]])

    ```
    """
    if vector.ndim != 1:
        msg = f"Expected a vector but received a {vector.ndim}-d tensor"
        raise ValueError(msg)
    n = vector.shape[0]
    return (
        torch.cat(
            [vector.flip((0,)), torch.narrow(vector.flip((0,)), dim=0, start=0, length=n - 1)],
            dim=0,
        )
        .unfold(0, n, 1)
        .flip((1,))
    )
