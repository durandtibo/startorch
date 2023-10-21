from __future__ import annotations

__all__ = ["check_num_examples", "check_std"]

from typing import Any


def check_num_examples(value: Any, name: str = "num_examples") -> None:
    r"""Checks if the given value is a valid number of examples.

    Args:
    ----
        value: Specifies the value to check.
        name (str, optional): Specifies the variable name.
            Default: ``'num_examples'``

    Raises:
    ------
        TypeError if the input is not an integer.
        RuntimeError if the value is not greater than 0
    """
    if not isinstance(value, int):
        raise TypeError(
            f"Incorrect type for {name}. Expected an integer but received {type(value)}"
        )
    if value < 1:
        raise RuntimeError(
            f"Incorrect value for {name}. Expected a value greater than 0 " f"but received {value}"
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
    """
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Incorrect type for {name}. Expected an integer or float but received {type(value)}"
        )
    if value < 0:
        raise RuntimeError(
            f"Incorrect value for {name}. Expected a value greater than 0 but received {value}"
        )
