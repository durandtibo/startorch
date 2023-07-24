from startorch.utils.conversion import to_tuple

##############################
#     Tests for to_tuple     #
##############################


def test_to_tuples_tuple() -> None:
    assert to_tuple((1, 2, 3)) == (1, 2, 3)


def test_to_tuples_list() -> None:
    assert to_tuple([1, 2, 3]) == (1, 2, 3)


def test_to_tuples_bool() -> None:
    assert to_tuple(True) == (True,)


def test_to_tuples_int() -> None:
    assert to_tuple(1) == (1,)


def test_to_tuples_float() -> None:
    assert to_tuple(42.1) == (42.1,)


def test_to_tuples_str() -> None:
    assert to_tuple("abc") == ("abc",)
