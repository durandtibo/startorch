r"""Contain the implementation of tensor transformers that computes
arithmetic functions on a tensor."""

from __future__ import annotations

__all__ = [
    "AbsTensorTransformer",
    "ClampTensorTransformer",
    "ExpTensorTransformer",
    "LogTensorTransformer",
]

from typing import TYPE_CHECKING

from startorch.tensor.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:

    import torch


class AbsTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the absolute value
    of a tensor.

    This tensor transformer is equivalent to: ``output = abs(input)``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Abs
    >>> transformer = Abs()
    >>> transformer
    AbsTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[1., 2., 3.], [4., 5., 6.]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.abs()


class ClampTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer to generate a tensor where the
    values are clamped.

    Note: ``min`` and ``max`` cannot be both ``None``.

    Args:
        min: The lower bound. If ``min`` is ``None``, there is no
            lower bound.
        max: The upper bound. If ``max`` is  ``None``, there is no
            upper bound.

    Raises:
        ValueError: if both ``min`` and ``max`` are ``None``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Clamp
    >>> transformer = Clamp(min=-2.0, max=2.0)
    >>> transformer
    ClampTensorTransformer(min=-2.0, max=2.0)
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[ 1., -2.,  2.], [-2.,  2., -2.]])

    ```
    """

    def __init__(
        self,
        min: float | None,  # noqa: A002
        max: float | None,  # noqa: A002
    ) -> None:
        if min is None and max is None:
            msg = "`min` and `max` cannot be both None"
            raise ValueError(msg)
        self._min = min
        self._max = max

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(min={self._min}, max={self._max})"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.clamp(self._min, self._max)


class ExpTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the exponential of
    the input tensor.

    This tensor transformer is equivalent to: ``output = exp(input)``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Exp
    >>> transformer = Exp()
    >>> transformer
    ExpTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[2.7183e+00, 1.3534e-01, 2.0086e+01], [1.8316e-02, 1.4841e+02, 2.4788e-03]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.exp()


class LogTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the logarithm of the
    input tensor.

    This tensor transformer is equivalent to: ``output = log(input)``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Log
    >>> transformer = Log()
    >>> transformer
    LogTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    >>> out
    tensor([[0.0000, 0.6931, 1.0986], [1.3863, 1.6094, 1.7918]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.log()
