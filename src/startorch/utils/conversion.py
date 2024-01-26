r"""Contain utility functions to convert objects."""

from __future__ import annotations

__all__ = ["to_tuple"]

from typing import Any


def to_tuple(value: Any) -> tuple:
    r"""Convert a value to a tuple.

    This function is a no-op if the input is a tuple.

    Args:
        value: Specifies the value to convert.

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
