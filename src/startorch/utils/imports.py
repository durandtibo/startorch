r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_iden",
    "check_matplotlib",
    "check_objectory",
    "check_plotly",
    "iden_available",
    "is_iden_available",
    "is_matplotlib_available",
    "is_objectory_available",
    "is_plotly_available",
    "matplotlib_available",
    "objectory_available",
    "plotly_available",
]

from typing import TYPE_CHECKING, Any

from coola.utils import package_available
from coola.utils.imports import decorator_package_available

if TYPE_CHECKING:
    from collections.abc import Callable

################
#     iden     #
################


def check_iden() -> None:
    r"""Check if the ``iden`` package is installed.

    Raises:
        RuntimeError: if the ``iden`` package is not installed.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import check_iden
    >>> check_iden()

    ```
    """
    if not is_iden_available():
        msg = (
            "'iden' package is required but not installed. "
            "You can install 'iden' package with the command:\n\n"
            "pip install iden\n"
        )
        raise RuntimeError(msg)


def is_iden_available() -> bool:
    r"""Indicate if the ``iden`` package is installed or not.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import is_iden_available
    >>> is_iden_available()

    ```
    """
    return package_available("iden")


def iden_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``iden``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``iden`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import iden_available
    >>> @iden_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_iden_available)


######################
#     matplotlib     #
######################


def check_matplotlib() -> None:
    r"""Check if the ``matplotlib`` package is installed.

    Raises:
        RuntimeError: if the ``matplotlib`` package is not installed.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import check_matplotlib
    >>> check_matplotlib()

    ```
    """
    if not is_matplotlib_available():
        msg = (
            "'matplotlib' package is required but not installed. "
            "You can install 'matplotlib' package with the command:\n\n"
            "pip install matplotlib\n"
        )
        raise RuntimeError(msg)


def is_matplotlib_available() -> bool:
    r"""Indicate if the ``matplotlib`` package is installed or not.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import is_matplotlib_available
    >>> is_matplotlib_available()

    ```
    """
    return package_available("matplotlib")


def matplotlib_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if
    ``matplotlib`` package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``matplotlib`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import matplotlib_available
    >>> @matplotlib_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_matplotlib_available)


#####################
#     objectory     #
#####################


def check_objectory() -> None:
    r"""Check if the ``objectory`` package is installed.

    Raises:
        RuntimeError: if the ``objectory`` package is not installed.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import check_objectory
    >>> check_objectory()

    ```
    """
    if not is_objectory_available():
        msg = (
            "'objectory' package is required but not installed. "
            "You can install 'objectory' package with the command:\n\n"
            "pip install objectory\n"
        )
        raise RuntimeError(msg)


def is_objectory_available() -> bool:
    r"""Indicate if the ``objectory`` package is installed or not.

    Returns:
        ``True`` if ``objectory`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import is_objectory_available
    >>> is_objectory_available()

    ```
    """
    return package_available("objectory")


def objectory_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``objectory``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``objectory`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import objectory_available
    >>> @objectory_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_objectory_available)


##################
#     plotly     #
##################


def check_plotly() -> None:
    r"""Check if the ``plotly`` package is installed.

    Raises:
        RuntimeError: if the ``plotly`` package is not installed.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import check_plotly
    >>> check_plotly()

    ```
    """
    if not is_plotly_available():
        msg = (
            "'plotly' package is required but not installed. "
            "You can install 'plotly' package with the command:\n\n"
            "pip install plotly\n"
        )
        raise RuntimeError(msg)


def is_plotly_available() -> bool:
    r"""Indicate if the ``plotly`` package is installed or not.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import is_plotly_available
    >>> is_plotly_available()

    ```
    """
    return package_available("plotly")


def plotly_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``plotly``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``plotly`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from startorch.utils.imports import plotly_available
    >>> @plotly_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_plotly_available)
