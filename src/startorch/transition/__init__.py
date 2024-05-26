r"""Contain transition matrix generators."""

from __future__ import annotations

__all__ = ["BaseTransitionGenerator", "Diagonal", "DiagonalTransitionGenerator"]

from startorch.transition.base import BaseTransitionGenerator
from startorch.transition.diag import DiagonalTransitionGenerator
from startorch.transition.diag import DiagonalTransitionGenerator as Diagonal
