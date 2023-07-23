from __future__ import annotations

__all__ = ["shapes_are_equal"]

from collections.abc import Sequence

from torch import Tensor


def shapes_are_equal(tensors: Sequence[Tensor]) -> bool:
    r"""Indicates if the shapes of several tensors are equal or not.

    This method does not check the values or the data type of the
    tensors.

    Args:
    ----
        tensors (sequence): Specifies the tensors to check.

    Returns:
    -------
        bool: ``True`` if all the tensors have the same shape,
            otherwise ``False``. By design, this function returns
            ``False`` if no tensor is provided.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.utils.tensor import shapes_are_equal
        >>> shapes_are_equal([torch.rand(2, 3), torch.rand(2, 3)])
        True
        >>> shapes_are_equal([torch.rand(2, 3), torch.rand(2, 3, 1)])
        False
    """
    if not tensors:
        return False
    shape = tensors[0].shape
    return all(shape == tensor.shape for tensor in tensors[1:])
