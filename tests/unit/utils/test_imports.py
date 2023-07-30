from unittest.mock import patch

from pytest import raises

from startorch.utils.imports import check_matplotlib, is_matplotlib_available

################################
#     Tests for matplotlib     #
################################


def test_check_matplotlib_with_package() -> None:
    with patch("startorch.utils.imports.is_matplotlib_available", lambda *args: True):
        check_matplotlib()


def test_check_matplotlib_without_package() -> None:
    with patch("startorch.utils.imports.is_matplotlib_available", lambda *args: False):
        with raises(RuntimeError, match="`matplotlib` package is required but not installed."):
            check_matplotlib()


def test_is_matplotlib_available() -> None:
    assert isinstance(is_matplotlib_available(), bool)
