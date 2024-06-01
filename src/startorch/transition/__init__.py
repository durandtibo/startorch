r"""Contain transition matrix generators."""

from __future__ import annotations

__all__ = [
    "BaseTransitionGenerator",
    "Diagonal",
    "DiagonalTransitionGenerator",
    "Mask",
    "MaskTransitionGenerator",
    "Normalize",
    "NormalizeTransitionGenerator",
    "PermuteDiagonal",
    "PermuteDiagonalTransitionGenerator",
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
from startorch.transition.diag import PermuteDiagonalTransitionGenerator
from startorch.transition.diag import (
    PermuteDiagonalTransitionGenerator as PermuteDiagonal,
)
from startorch.transition.mask import MaskTransitionGenerator
from startorch.transition.mask import MaskTransitionGenerator as Mask
from startorch.transition.normalize import NormalizeTransitionGenerator
from startorch.transition.normalize import NormalizeTransitionGenerator as Normalize
from startorch.transition.tensor import TensorTransitionGenerator
from startorch.transition.transform import TransformTransitionGenerator
from startorch.transition.transform import TransformTransitionGenerator as Transform
