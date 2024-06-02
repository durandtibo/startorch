r"""Contain the implementation of a generic tensor transformer."""

from __future__ import annotations

__all__ = ["AddTransformer"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from startorch.transformer.base import BaseTransformer
from startorch.transformer.utils import add_item, check_input_keys

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import torch

logger = logging.getLogger(__name__)


class AddTransformer(BaseTransformer):
    r"""Implements a tensor transformer that adds multiple tensors.

    Args:
        inputs: The keys to add.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import AddTransformer
    >>> transformer = AddTransformer(inputs=["input1", "input2"], output="output")
    >>> transformer
    AddTransformer(
      (inputs): ('input1', 'input2')
      (output): output
      (exist_ok): False
    )
    >>> data = {
    ...     "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
    ...     "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ... }
    >>> out = transformer.transform(data)
    >>> out
    {'input1': tensor([[ 0., -1.,  2.], [-4.,  5., -6.]]),
     'input2': tensor([[1., 2., 3.], [4., 5., 6.]]),
     'output': tensor([[ 1.,  1.,  5.], [ 0., 10.,  0.]])}

    ```
    """

    def __init__(
        self,
        inputs: Sequence[str],
        output: str,
        exist_ok: bool = False,
    ) -> None:
        if not inputs:
            msg = r"inputs cannot be empty"
            raise ValueError(msg)
        self._inputs = tuple(inputs)
        self._output = output
        self._exist_ok = exist_ok

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "inputs": self._inputs,
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
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> dict[Hashable, torch.Tensor]:
        check_input_keys(data, keys=self._inputs)
        data = data.copy()
        value = data[self._inputs[0]].clone()
        for key in self._inputs[1:]:
            value += data[key]
        add_item(data, key=self._output, value=value, exist_ok=self._exist_ok)
        return data
