from unittest.mock import patch

import pytest

from startorch.utils.imports import (
    check_iden,
    check_matplotlib,
    check_objectory,
    check_plotly,
    is_iden_available,
    is_matplotlib_available,
    is_objectory_available,
    is_plotly_available,
    objectory_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


##########################
#     Tests for iden     #
##########################


def test_check_iden_with_package() -> None:
    with patch("startorch.utils.imports.is_iden_available", lambda: True):
        check_iden()


def test_check_iden_without_package() -> None:
    with (
        patch("startorch.utils.imports.is_iden_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'iden' package is required but not installed."),
    ):
        check_iden()


def test_is_iden_available() -> None:
    assert isinstance(is_iden_available(), bool)


################################
#     Tests for matplotlib     #
################################


def test_check_matplotlib_with_package() -> None:
    with patch("startorch.utils.imports.is_matplotlib_available", lambda: True):
        check_matplotlib()


def test_check_matplotlib_without_package() -> None:
    with (
        patch("startorch.utils.imports.is_matplotlib_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'matplotlib' package is required but not installed."),
    ):
        check_matplotlib()


def test_is_matplotlib_available() -> None:
    assert isinstance(is_matplotlib_available(), bool)


#####################
#     objectory     #
#####################


def test_check_objectory_with_package() -> None:
    with patch("startorch.utils.imports.is_objectory_available", lambda: True):
        check_objectory()


def test_check_objectory_without_package() -> None:
    with (
        patch("startorch.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'objectory' package is required but not installed."),
    ):
        check_objectory()


def test_is_objectory_available() -> None:
    assert isinstance(is_objectory_available(), bool)


def test_objectory_available_with_package() -> None:
    with patch("startorch.utils.imports.is_objectory_available", lambda: True):
        fn = objectory_available(my_function)
        assert fn(2) == 44


def test_objectory_available_without_package() -> None:
    with patch("startorch.utils.imports.is_objectory_available", lambda: False):
        fn = objectory_available(my_function)
        assert fn(2) is None


def test_objectory_available_decorator_with_package() -> None:
    with patch("startorch.utils.imports.is_objectory_available", lambda: True):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_objectory_available_decorator_without_package() -> None:
    with patch("startorch.utils.imports.is_objectory_available", lambda: False):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


############################
#     Tests for plotly     #
############################


def test_check_plotly_with_package() -> None:
    with patch("startorch.utils.imports.is_plotly_available", lambda: True):
        check_plotly()


def test_check_plotly_without_package() -> None:
    with (
        patch("startorch.utils.imports.is_plotly_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'plotly' package is required but not installed."),
    ):
        check_plotly()


def test_is_plotly_available() -> None:
    assert isinstance(is_plotly_available(), bool)
