r"""Contain some testing utility features like pytest fixtures."""

from __future__ import annotations

__all__ = ["iden_available", "matplotlib_available", "plotly_available"]

import pytest

from startorch.utils.imports import (
    is_iden_available,
    is_matplotlib_available,
    is_plotly_available,
)

iden_available = pytest.mark.skipif(not is_iden_available(), reason="Requires iden")
matplotlib_available = pytest.mark.skipif(
    not is_matplotlib_available(), reason="Requires matplotlib"
)
plotly_available = pytest.mark.skipif(not is_plotly_available(), reason="Requires plotly")
