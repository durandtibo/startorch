r"""Contain transition matrix generators."""

from __future__ import annotations

__all__ = [
    "BaseTransitionGenerator",
    "Diagonal",
    "DiagonalTransitionGenerator",
    "is_transition_generator_config",
    "setup_transition_generator",
]

from startorch.transition.base import (
    BaseTransitionGenerator,
    is_transition_generator_config,
    setup_transition_generator,
)
from startorch.transition.diag import DiagonalTransitionGenerator
from startorch.transition.diag import DiagonalTransitionGenerator as Diagonal
