from typing import Any

from pytest import mark, raises

from startorch.example.utils import check_interval, check_num_examples, check_std

####################################
#     Tests for check_interval     #
####################################


@mark.parametrize("value", (0, 1, 1.2, 2, 2.9))
def test_check_interval_valid(value: int) -> None:
    check_interval(value, low=0.0, high=3.0, name="my_variable")


@mark.parametrize("value", ("abc", None))
def test_check_interval_incorrect_type(value: Any) -> None:
    with raises(
        TypeError, match="Incorrect type for my_variable. Expected an integer or float but received"
    ):
        check_interval(value, low=0.0, high=3.0, name="my_variable")


@mark.parametrize("value", (-1, -0.01, 3, 4.2))
def test_check_interval_incorrect_value(value: int) -> None:
    with raises(
        RuntimeError, match="Incorrect value for my_variable. Expected a value in interval"
    ):
        check_interval(value, low=0.0, high=3.0, name="my_variable")


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


def test_check_num_examples_incorrect_type_custom_name() -> None:
    with raises(TypeError, match="Incorrect type for n_samples. Expected an integer but received"):
        check_num_examples(1.2, name="n_samples")


@mark.parametrize("value", (0, -1))
def test_check_num_examples_incorrect_value(value: int) -> None:
    with raises(
        RuntimeError, match="Incorrect value for num_examples. Expected a value greater than 0"
    ):
        check_num_examples(value)


def test_check_num_examples_incorrect_value_custom_name() -> None:
    with raises(
        RuntimeError, match="Incorrect value for n_samples. Expected a value greater than 0"
    ):
        check_num_examples(0, "n_samples")


###############################
#     Tests for check_std     #
###############################


@mark.parametrize("value", (1, 1.2, 2))
def test_check_std_valid(value: int) -> None:
    check_std(value)


@mark.parametrize("value", ("abc", None))
def test_check_std_incorrect_type(value: Any) -> None:
    with raises(
        TypeError, match="Incorrect type for std. Expected an integer or float but received"
    ):
        check_std(value)


def test_check_std_incorrect_type_custom_name() -> None:
    with raises(
        TypeError,
        match="Incorrect type for noise_std. Expected an integer or float but received",
    ):
        check_std("-1", name="noise_std")


@mark.parametrize("value", (-1, -4.2))
def test_check_std_incorrect_value(value: int) -> None:
    with raises(RuntimeError, match="Incorrect value for std. Expected a value greater than 0"):
        check_std(value)


def test_check_std_incorrect_value_custom_name() -> None:
    with raises(
        RuntimeError, match="Incorrect value for noise_std. Expected a value greater than 0"
    ):
        check_std(-1, name="noise_std")
