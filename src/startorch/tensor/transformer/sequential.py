r"""Contain the implementation of a transformer that sequentially
computes tensor transformations."""

from __future__ import annotations

__all__ = ["SequentialTensorTransformer"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_sequence

from startorch.tensor.transformer.base import (
    BaseTensorTransformer,
    setup_tensor_transformer,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

logger = logging.getLogger(__name__)


class SequentialTensorTransformer(BaseTensorTransformer):
    r"""Implement a transformer that sequentially computes tensor
    transformations.

    Args:
        transformers: The sequence of tensor transformers or their
            configurations.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Sequential, Abs, Clamp
    >>> transformer = Sequential([Abs(), Clamp(min=-1, max=2)])
    >>> transformer
    SequentialTensorTransformer(
      (0): AbsTensorTransformer()
      (1): ClampTensorTransformer(min=-1, max=2)
    )
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[1., 2., 2.], [2., 2., 2.]])

    ```
    """

    def __init__(self, transformers: Sequence[BaseTensorTransformer | dict]) -> None:
        self._transformers = [setup_tensor_transformer(transformer) for transformer in transformers]

    def __repr__(self) -> str:
        args = repr_indent(repr_sequence(self._transformers))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,
    ) -> torch.Tensor:
        for transformer in self._transformers:
            tensor = transformer.transform(tensor, rng=rng)
        return tensor
