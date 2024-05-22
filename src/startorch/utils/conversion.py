r"""Contain utility functions to convert objects."""

from __future__ import annotations

__all__ = ["to_array", "to_tensor", "to_tuple"]


from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


def to_array(data: Sequence | torch.Tensor | np.ndarray) -> np.ndarray:
    r"""Convert the input to a ``numpy.ndarray``.

    Args:
        data: The data to convert to an array.

    Returns:
        A NumPy array.

    Example usage:

    ```pycon

    >>> from startorch.utils.conversion import to_array
    >>> x = to_array([1, 2, 3, 4, 5])
    >>> x
    array([1, 2, 3, 4, 5])

    ```
    """
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.numpy()
    return np.asarray(data)


def to_tensor(data: torch.Tensor | np.ndarray | Sequence) -> torch.Tensor:
    r"""Convert the input to a ``torch.Tensor``.

    Args:
        data: The data to convert to a tensor.

    Returns:
        A tensor.

    Example usage:

    ```pycon

    >>> from startorch.utils.conversion import to_tensor
    >>> x = to_tensor([1, 2, 3, 4, 5])
    >>> x
    tensor([1, 2, 3, 4, 5])

    ```
    """
    if torch.is_tensor(data):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return torch.as_tensor(data)


def to_tuple(value: Any) -> tuple:
    r"""Convert a value to a tuple.

    This function is a no-op if the input is a tuple.

    Args:
        value: The value to convert.

    Returns:
        The input value in a tuple.

    Example usage:

    ```pycon

    >>> from startorch.utils.conversion import to_tuple
    >>> to_tuple(1)
    (1,)
    >>> to_tuple("abc")
    ('abc',)

    ```
    """
    if isinstance(value, tuple):
        return value
    if isinstance(value, (bool, int, float, str)):
        return (value,)
    return tuple(value)
