r"""Contain tensor transformers."""

from __future__ import annotations

__all__ = [
    "BaseTensorTransformer",
    "Identity",
    "IdentityTensorTransformer",
    "is_tensor_transformer_config",
    "setup_tensor_transformer",
]

from startorch.transformer.tensor.base import (
    BaseTensorTransformer,
    is_tensor_transformer_config,
    setup_tensor_transformer,
)
from startorch.transformer.tensor.identity import IdentityTensorTransformer
from startorch.transformer.tensor.identity import IdentityTensorTransformer as Identity
