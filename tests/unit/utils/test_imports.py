from unittest.mock import patch

import pytest

from startorch.utils.imports import (
    check_iden,
    check_matplotlib,
    check_plotly,
    is_iden_available,
    is_matplotlib_available,
    is_plotly_available,
)

################################
#     Tests for iden     #
################################


def test_check_iden_with_package() -> None:
    with patch("startorch.utils.imports.is_iden_available", lambda: True):
        check_iden()


def test_check_iden_without_package() -> None:
    with (
        patch("startorch.utils.imports.is_iden_available", lambda: False),
        pytest.raises(RuntimeError, match="`iden` package is required but not installed."),
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
        pytest.raises(RuntimeError, match="`matplotlib` package is required but not installed."),
    ):
        check_matplotlib()


def test_is_matplotlib_available() -> None:
    assert isinstance(is_matplotlib_available(), bool)


############################
#     Tests for plotly     #
############################


def test_check_plotly_with_package() -> None:
    with patch("startorch.utils.imports.is_plotly_available", lambda: True):
        check_plotly()


def test_check_plotly_without_package() -> None:
    with (
        patch("startorch.utils.imports.is_plotly_available", lambda: False),
        pytest.raises(RuntimeError, match="`plotly` package is required but not installed."),
    ):
        check_plotly()


def test_is_plotly_available() -> None:
    assert isinstance(is_plotly_available(), bool)
