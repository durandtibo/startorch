r"""Contain utility functions."""

from __future__ import annotations

__all__ = ["add_item", "check_input_keys"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


def add_item(data: dict, key: Hashable, value: Any, exist_ok: bool = False) -> None:
    r"""Add an item to a dictionary.

    Args:
        data: The dictionary where to add the item.
        key: The key to add to the dictionary.
        value: The value to add to the dictionary.
        exist_ok: If ``False``, an exception is raised if the key
            already exists. Otherwise, the value associated to the
            key is updated.

    Raises:
        KeyError: If the key already exists and ``exist_ok=False``.
    """
    if key in data and not exist_ok:
        msg = f"Key {key} already exists. Please use exist_ok=True to overwrite this key"
        raise KeyError(msg)
    data[key] = value


def check_input_keys(data: dict, keys: Sequence[Hashable]) -> None:
    r"""Check if the keys exist.

    Args:
        data: The dictionary of data.
        keys: The keys to check.

    Raises:
        KeyError: if a key is missing.
    """
    for key in keys:
        if key not in data:
            msg = f"Missing key: {key}. The available keys are: {sorted(data.keys())}"
            raise KeyError(msg)
