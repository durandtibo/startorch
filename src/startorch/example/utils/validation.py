from __future__ import annotations

__all__ = [
    "check_interval",
    "check_num_examples",
    "check_positive_integer",
    "check_std",
]

from typing import Any


def check_interval(value: Any, low: float, high: float, name: str) -> None:
    r"""Checks if the given value is an interval.

    Args:
    ----
        value: Specifies the value to check.
        low (float): Specifies the minimum value (inclusive).
        high (float): Specifies the maximum value (exclusive).
        name (str): Specifies the variable name.

    Raises:
    ------
        TypeError if the input is not an integer or float.
        RuntimeError if the value is not in the interval

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.example.utils import check_interval
        >>> check_interval(1, low=-1.0, high=2.0, name="my_variable")
    """
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Incorrect type for {name}. Expected an integer or float but received {type(value)}"
        )
    if value < low or value >= high:
        raise RuntimeError(
            f"Incorrect value for {name}. Expected a value in interval [{low}, {high}) "
            f"but received {value}"
        )


def check_num_examples(value: Any) -> None:
    r"""Checks if the given value is a valid number of examples.

    Args:
    ----
        value: Specifies the value to check.

    Raises:
    ------
        TypeError if the input is not an integer.
        RuntimeError if the value is not greater than 0

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.example.utils import check_num_examples
        >>> check_num_examples(5)
    """
    check_positive_integer(value, name="num_examples")


def check_positive_integer(value: Any, name: str) -> None:
    r"""Checks if the given value is a valid positive integer.

    Args:
    ----
        value: Specifies the value to check.
        name (str): Specifies the variable name.

    Raises:
    ------
        TypeError if the input is not an integer.
        RuntimeError if the value is not greater than 0

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.example.utils import check_positive_integer
        >>> check_positive_integer(5, name='feature_size')
    """
    if not isinstance(value, int):
        raise TypeError(
            f"Incorrect type for {name}. Expected an integer but received {type(value)}"
        )
    if value < 1:
        raise RuntimeError(
            f"Incorrect value for {name}. Expected a value greater than 0 but received {value}"
        )


def check_std(value: Any, name: str = "std") -> None:
    r"""Checks if the given value is a valid standard deviation.

    Args:
    ----
        value: Specifies the value to check.
        name (str, optional): Specifies the variable name.
            Default: ``'std'``

    Raises:
    ------
        TypeError if the input is not an integer or float.
        RuntimeError if the value is not greater than 0

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.example.utils import check_std
        >>> check_std(1.2)
    """
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Incorrect type for {name}. Expected an integer or float but received {type(value)}"
        )
    if value < 0:
        raise RuntimeError(
            f"Incorrect value for {name}. Expected a value greater than 0 but received {value}"
        )
