from __future__ import annotations

__all__ = ["check_num_examples"]


def check_num_examples(value: int) -> None:
    r"""Checks if the given value is a valid number of examples.

    Args:
    ----
        num_examples (int): Specifies the value to check.

    Raises:
    ------
        TypeError if the input is not an integer.
        RuntimeError if the value is not greater than 0
    """
    if not isinstance(value, int):
        raise TypeError(
            f"Incorrect type for num_examples. Expected an integer but received {type(value)}"
        )
    if value < 1:
        raise RuntimeError(
            f"Incorrect value for num_examples. Expected a value greater than 0 "
            f"but received {value}"
        )
