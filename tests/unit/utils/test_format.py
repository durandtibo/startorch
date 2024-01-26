from __future__ import annotations

import pytest
from objectory import OBJECT_TARGET

from startorch.utils.format import str_target_object, str_weighted_modules

#######################################
#     Tests for str_target_object     #
#######################################


def test_str_target_object_with_target() -> None:
    assert (
        str_target_object({OBJECT_TARGET: "something.MyClass"}) == "[_target_: something.MyClass]"
    )


def test_str_target_object_without_target() -> None:
    assert str_target_object({}) == "[_target_: N/A]"


##########################################
#     Tests for str_weighted_modules     #
##########################################


def test_str_weighted_modules_empty() -> None:
    assert str_weighted_modules([], []) == ""


def test_str_weighted_modules_1() -> None:
    assert str_weighted_modules(["abc"], [2]) == "(0) [weight=2] abc"


def test_str_weighted_modules_2() -> None:
    assert (
        str_weighted_modules(["abc", "something\nelse"], [2, 1])
        == "(0) [weight=2] abc\n(1) [weight=1] something\n  else"
    )


def test_str_weighted_modules_different_lengths() -> None:
    with pytest.raises(RuntimeError, match="`modules` and `weights` must have the same length"):
        str_weighted_modules(["abc"], [2, 3])
