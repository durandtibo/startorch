from __future__ import annotations

__all__ = ["matplotlib_available"]

from pytest import mark

from startorch.utils.imports import is_matplotlib_available

matplotlib_available = mark.skipif(not is_matplotlib_available(), reason="Requires matplotlib")
