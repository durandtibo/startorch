from __future__ import annotations

from objectory import OBJECT_TARGET

from startorch.utils.format import str_target_object

#######################################
#     Tests for str_target_object     #
#######################################


def test_str_target_object_with_target() -> None:
    assert (
        str_target_object({OBJECT_TARGET: "something.MyClass"}) == "[_target_: something.MyClass]"
    )


def test_str_target_object_without_target() -> None:
    assert str_target_object({}) == "[_target_: N/A]"
