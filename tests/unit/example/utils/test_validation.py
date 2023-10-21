from typing import Any

from pytest import mark, raises

from startorch.example.utils import check_num_examples

########################################
#     Tests for check_num_examples     #
########################################


@mark.parametrize("value", (1, 2, 100))
def test_check_num_examples_valid(value: int) -> None:
    check_num_examples(value)


@mark.parametrize("value", (1.2, "abc", None))
def test_check_num_examples_incorrect_type(value: Any) -> None:
    with raises(
        TypeError, match="Incorrect type for num_examples. Expected an integer but received"
    ):
        check_num_examples(value)


@mark.parametrize("value", (0, -1))
def test_check_num_examples_incorrect_value(value: int) -> None:
    with raises(
        RuntimeError, match="Incorrect value for num_examples. Expected a value greater than 0"
    ):
        check_num_examples(value)
