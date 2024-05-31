r"""Contain the implementation a transformer that sequentially computes
transformations."""

from __future__ import annotations

__all__ = ["SequentialTransformer"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_sequence

from startorch.transformer.base import BaseTransformer, setup_transformer

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import torch

logger = logging.getLogger(__name__)


class SequentialTransformer(BaseTransformer):
    r"""Implement a transformer that sequentially computes
    transformations.

    Args:
        transformers: The sequence of transformers or their
            configurations.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Sequential, TensorTransformer
    >>> from startorch.tensor.transformer import Abs, Clamp
    >>> transformer = Sequential(
    ...     [
    ...         TensorTransformer(transformer=Abs(), input="input", output="output1"),
    ...         TensorTransformer(
    ...             transformer=Clamp(min=-1, max=2), input="input", output="output2"
    ...         ),
    ...     ]
    ... )
    >>> transformer
    SequentialTransformer(
      (0): TensorTransformer(
          (transformer): AbsTensorTransformer()
          (input): input
          (output): output1
          (exist_ok): False
        )
      (1): TensorTransformer(
          (transformer): ClampTensorTransformer(min=-1, max=2)
          (input): input
          (output): output2
          (exist_ok): False
        )
    )
    >>> data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'input': tensor([[ 1., -2.,  3.],
                      [-4.,  5., -6.]]),
     'output1': tensor([[1., 2., 3.],
                        [4., 5., 6.]]),
     'output2': tensor([[ 1., -1.,  2.],
                        [-1.,  2., -1.]])}

    ```
    """

    def __init__(self, transformers: Sequence[BaseTransformer | dict]) -> None:
        self._transformers = [setup_transformer(transformer) for transformer in transformers]

    def __repr__(self) -> str:
        args = repr_indent(repr_sequence(self._transformers))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def transform(
        self,
        data: dict[Hashable, torch.Tensor],
        *,
        rng: torch.Transformer | None = None,
    ) -> dict[Hashable, torch.Tensor]:
        for transformer in self._transformers:
            data = transformer.transform(data, rng=rng)
        return data
