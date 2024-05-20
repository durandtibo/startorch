r"""Contain the implementation of tensor transformers that computes
arithmetic functions on tensors."""

from __future__ import annotations

__all__ = [
    "AbsTransformer",
    "ClampTransformer",
]

from typing import TYPE_CHECKING

from startorch.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:

    import torch


class AbsTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the absolute value
    of a tensor.

    This tensor transformer is equivalent to: ``output = abs(input)``

    Args:
        input: The key that contains the input tensor.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Abs
    >>> transformer = Abs(input="input", output="output")
    >>> transformer
    AbsTransformer(input=input, output=output, exist_ok=False)
    >>> data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'input': tensor([[ 1., -2.,  3.],
                      [-4.,  5., -6.]]),
     'output': tensor([[1., 2., 3.],
                       [4., 5., 6.]])}

    ```
    """

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.abs()


class ClampTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer to generate tensors where the
    values are clamped.

    Note: ``min`` and ``max`` cannot be both ``None``.

    Args:
        input: The key that contains the input tensor.
        output: The key that contains the output tensor.
        min: The lower bound. If ``min`` is ``None``, there is no
            lower bound.
        max: The upper bound. If ``max`` is  ``None``, there is no
            upper bound.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Raises:
        ValueError: if both ``min`` and ``max`` are ``None``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Clamp
    >>> transformer = Clamp(input="input", output="output", min=-2.0, max=2.0)
    >>> transformer
    ClampTransformer(input=input, output=output, min=-2.0, max=2.0, exist_ok=False)
    >>> data = {"input": torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'input': tensor([[ 1., -2.,  3.],
                      [-4.,  5., -6.]]),
     'output': tensor([[ 1., -2.,  2.],
                       [-2.,  2., -2.]])}

    ```
    """

    def __init__(
        self,
        input: str,  # noqa: A002
        output: str,
        min: float | None,  # noqa: A002
        max: float | None,  # noqa: A002
        exist_ok: bool = False,
    ) -> None:
        super().__init__(input=input, output=output, exist_ok=exist_ok)
        if min is None and max is None:
            msg = "`min` and `max` cannot be both None"
            raise ValueError(msg)
        self._min = min
        self._max = max

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(input={self._input}, output={self._output}, "
            f"min={self._min}, max={self._max}, exist_ok={self._exist_ok})"
        )

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.clamp(self._min, self._max)
