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
from startorch.transition.diag import (
    DiagonalTransitionGenerator,
    DiagonalTransitionGenerator as Diagonal,
    PermuteDiagonalTransitionGenerator,
    PermuteDiagonalTransitionGenerator as PermuteDiagonal,
)
from startorch.transition.mask import MaskTransitionGenerator, MaskTransitionGenerator as Mask
from startorch.transition.normalize import (
    NormalizeTransitionGenerator,
    NormalizeTransitionGenerator as Normalize,
)
from startorch.transition.tensor import TensorTransitionGenerator
from startorch.transition.transform import (
    TransformTransitionGenerator,
    TransformTransitionGenerator as Transform,
)
