r"""Contain fallback types for ``iden``."""

from __future__ import annotations

__all__ = ["BaseDataGenerator"]

from typing import Generic, TypeVar

from startorch.utils.imports import check_iden

T = TypeVar("T")


class BaseDataGenerator(Generic[T]):
    r"""Implement a fallback object for ``BaseDataGenerator`` if ``iden``
    is not installed.

    This class allows the code to run even when ``iden`` is missing,
    but any attempt to instantiate a ``BaseDataGenerator`` object will
    raise an error.
    """

    def __init__(self) -> None:
        super().__init__()
        check_iden()
