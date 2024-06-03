r"""Contain the implementation of a generic tensor transformer."""

from __future__ import annotations

__all__ = ["AddTransformer", "DivTransformer", "MulTransformer", "SubTransformer"]

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

    Raises:
        ValueError: if ``inputs`` is empty.

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


class DivTransformer(BaseTransformer):
    r"""Implements a tensor transformer that computes the division
    between two tensors.

    This transformer is equivalent to:
    ``output = dividend / divisor``

    Args:
        dividend: The key that contains the dividend.
        divisor: The key that contains the divisor.
        output: The key that contains the output tensor.
        rounding_mode: The
            type of rounding applied to the result.
            - ``None``: true division.
            - ``"trunc"``: rounds the results of the division
                towards zero.
            - ``"floor"``: floor division.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import DivTransformer
    >>> transformer = DivTransformer(dividend="input1", divisor="input2", output="output")
    >>> transformer
    DivTransformer(dividend=input1, divisor=input2, output=output, rounding_mode=None, exist_ok=False)
    >>> data = {
    ...     "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
    ...     "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ... }
    >>> out = transformer.transform(data)
    >>> out
    {'input1': tensor([[ 0., -1.,  2.], [-4.,  5., -6.]]),
     'input2': tensor([[1., 2., 3.], [4., 5., 6.]]),
     'output': tensor([[ 0.0000, -0.5000,  0.6667], [-1.0000,  1.0000, -1.0000]])}

    ```
    """

    def __init__(
        self,
        dividend: str,
        divisor: str,
        output: str,
        rounding_mode: str | None = None,
        exist_ok: bool = False,
    ) -> None:
        self._dividend = dividend
        self._divisor = divisor
        self._output = output
        self._rounding_mode = rounding_mode
        self._exist_ok = exist_ok

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(dividend={self._dividend}, "
            f"divisor={self._divisor}, output={self._output}, "
            f"rounding_mode={self._rounding_mode}, exist_ok={self._exist_ok})"
        )

    def transform(
        self,
        data: dict[Hashable, torch.Tensor],
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> dict[Hashable, torch.Tensor]:
        check_input_keys(data, keys=[self._dividend, self._divisor])
        data = data.copy()
        add_item(
            data,
            key=self._output,
            value=data[self._dividend].div(data[self._divisor], rounding_mode=self._rounding_mode),
            exist_ok=self._exist_ok,
        )
        return data


class MulTransformer(BaseTransformer):
    r"""Implements a tensor transformer that multiplies multiple tensors.

    Args:
        inputs: The keys to multiply.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Raises:
        ValueError: if ``inputs`` is empty.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import MulTransformer
    >>> transformer = MulTransformer(inputs=["input1", "input2"], output="output")
    >>> transformer
    MulTransformer(
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
     'output': tensor([[  0.,  -2.,   6.], [-16.,  25., -36.]])}

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
            value *= data[key]
        add_item(data, key=self._output, value=value, exist_ok=self._exist_ok)
        return data


class SubTransformer(BaseTransformer):
    r"""Implements a tensor transformer that computes the difference
    between two tensors.

    This transformer is equivalent to:
    ``output = minuend - subtrahend``

    Args:
        minuend: The key that contains the minuend.
        subtrahend: The key that contains the subtrahend.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import SubTransformer
    >>> transformer = SubTransformer(minuend="input1", subtrahend="input2", output="output")
    >>> transformer
    SubTransformer(minuend=input1, subtrahend=input2, output=output, exist_ok=False)
    >>> data = {
    ...     "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
    ...     "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ... }
    >>> out = transformer.transform(data)
    >>> out
    {'input1': tensor([[ 0., -1.,  2.], [-4.,  5., -6.]]),
     'input2': tensor([[1., 2., 3.], [4., 5., 6.]]),
     'output': tensor([[ -1.,  -3.,  -1.], [ -8.,   0., -12.]])}

    ```
    """

    def __init__(
        self,
        minuend: str,
        subtrahend: str,
        output: str,
        exist_ok: bool = False,
    ) -> None:
        self._minuend = minuend
        self._subtrahend = subtrahend
        self._output = output
        self._exist_ok = exist_ok

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(minuend={self._minuend}, "
            f"subtrahend={self._subtrahend}, output={self._output}, exist_ok={self._exist_ok})"
        )

    def transform(
        self,
        data: dict[Hashable, torch.Tensor],
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> dict[Hashable, torch.Tensor]:
        check_input_keys(data, keys=[self._minuend, self._subtrahend])
        data = data.copy()
        add_item(
            data,
            key=self._output,
            value=data[self._minuend].sub(data[self._subtrahend]),
            exist_ok=self._exist_ok,
        )
        return data
