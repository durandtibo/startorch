r"""Contain utility functions for PyTorch tensors."""

from __future__ import annotations

__all__ = ["shapes_are_equal"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


def shapes_are_equal(tensors: Sequence[torch.Tensor]) -> bool:
    r"""Return ``True`` if the shapes of several tensors are equal,
    otherwise ``False``.

    This method does not check the values or the data type of the
    tensors.

    Args:
        tensors: Specifies the tensors to check.

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
