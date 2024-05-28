r"""Contain transition matrix generators."""

from __future__ import annotations

__all__ = [
    "BaseTransitionGenerator",
    "Diagonal",
    "DiagonalTransitionGenerator",
    "Masked",
    "MaskedTransitionGenerator",
    "Normalize",
    "NormalizeTransitionGenerator",
    "PermutedDiagonal",
    "PermutedDiagonalTransitionGenerator",
    "TensorTransitionGenerator",
    "Transform",
    "TransformTransitionGenerator",
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
from startorch.transition.diag import PermutedDiagonalTransitionGenerator
from startorch.transition.diag import (
    PermutedDiagonalTransitionGenerator as PermutedDiagonal,
)
from startorch.transition.mask import MaskedTransitionGenerator
from startorch.transition.mask import MaskedTransitionGenerator as Masked
from startorch.transition.normalize import NormalizeTransitionGenerator
from startorch.transition.normalize import NormalizeTransitionGenerator as Normalize
from startorch.transition.tensor import TensorTransitionGenerator
from startorch.transition.transform import TransformTransitionGenerator
from startorch.transition.transform import TransformTransitionGenerator as Transform
