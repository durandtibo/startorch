r"""Contain the implementation of a tensor identity transformer."""

from __future__ import annotations

__all__ = ["IdentityTensorTransformer"]

import logging
from typing import TYPE_CHECKING

from startorch.tensor.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class IdentityTensorTransformer(BaseTensorTransformer):
    r"""Implement the identity transformation.

    Args:
        copy: If ``True``, it returns a copy of the input tensor,
            otherwise it returns the input tensor.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Identity
    >>> transformer = Identity()
    >>> transformer
    IdentityTransformer(copy=True)
    >>> data = {"key": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'key': tensor([[1., 2., 3.],
                    [4., 5., 6.]])}

    ```
    """

    def __init__(self, copy: bool = True) -> None:
        super().__init__()
        self._copy = copy

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(copy={self._copy})"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        if self._copy:
            return tensor.clone()
        return tensor
