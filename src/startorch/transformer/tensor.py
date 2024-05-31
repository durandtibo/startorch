r"""Contain the implementation of a generic tensor transformer."""

from __future__ import annotations

__all__ = ["TensorTransformer"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from startorch.tensor.transformer.base import (
    BaseTensorTransformer,
    setup_tensor_transformer,
)
from startorch.transformer.base import BaseTransformer
from startorch.transformer.utils import add_item, check_input_keys

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch

logger = logging.getLogger(__name__)


class TensorTransformer(BaseTransformer):
    r"""Implements a generic tensor transformer.

    Args:
        transformer: The tensor transformer or its configuration.
        input: The key that contains the input tensor.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import TensorTransformer
    >>> from startorch.tensor.transformer import Abs
    >>> transformer = TensorTransformer(transformer=Abs(), input="input", output="output")
    >>> transformer
    TensorTransformer(
      (transformer): AbsTensorTransformer()
      (input): input
      (output): output
      (exist_ok): False
    )
    >>> data = {"input": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'input': tensor([[ 0., -1.,  2.],
                      [-4.,  5., -6.]]),
     'output': tensor([[0., 1., 2.],
                       [4., 5., 6.]])}

    ```
    """

    def __init__(
        self,
        transformer: BaseTensorTransformer | dict,
        input: str,  # noqa: A002
        output: str,
        exist_ok: bool = False,
    ) -> None:
        self._transformer = setup_tensor_transformer(transformer)
        self._input = input
        self._output = output
        self._exist_ok = exist_ok

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "transformer": self._transformer,
                    "input": self._input,
                    "output": self._output,
                    "exist_ok": self._exist_ok,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def transform(
        self,
        data: dict[Hashable, torch.Tensor],
        *,
        rng: torch.Transformer | None = None,
    ) -> dict[Hashable, torch.Tensor]:
        check_input_keys(data, keys=[self._input])
        data = data.copy()
        add_item(
            data,
            key=self._output,
            value=self._transformer.transform(data[self._input], rng=rng),
            exist_ok=self._exist_ok,
        )
        return data
