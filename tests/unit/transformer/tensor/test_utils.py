from __future__ import annotations

import pytest
from coola import objects_are_equal

from startorch.transformer.tensor.utils import add_item, check_input_keys

##############################
#     Tests for add_item     #
##############################


def test_add_item() -> None:
    data = {}
    add_item(data, "key", 1)
    assert objects_are_equal(data, {"key": 1})


def test_add_item_key_exist_ok_false() -> None:
    data = {"key": 0}
    with pytest.raises(KeyError):
        add_item(data, "key", 1)


def test_add_item_key_exist_ok_true() -> None:
    data = {"key": 0}
    add_item(data, "key", 1, exist_ok=True)
    assert objects_are_equal(data, {"key": 1})


######################################
#     Tests for check_input_keys     #
######################################


def test_check_input_keys_exist() -> None:
    check_input_keys(data={"key1": 1, "key2": 2, "key3": 3}, keys=["key1", "key3"])


def test_check_input_keys_missing() -> None:
    with pytest.raises(KeyError, match="Missing key: key4"):
        check_input_keys(data={"key1": 1, "key2": 2, "key3": 3}, keys=["key1", "key4"])


def test_check_input_keys_empty() -> None:
    check_input_keys(data={}, keys=[])
