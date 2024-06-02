r"""Contain the implementation of tensor transformers that computes
arithmetic functions on a tensor."""

from __future__ import annotations

__all__ = [
    "AbsTensorTransformer",
    "ClampTensorTransformer",
    "ExpTensorTransformer",
    "Expm1TensorTransformer",
    "LogTensorTransformer",
    "Log1pTensorTransformer",
    "PowTensorTransformer",
    "SqrtTensorTransformer",
    "CeilTensorTransformer",
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


class CeilTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the ceil of the
    elements of the input tensor.

    This tensor transformer is equivalent to: ``output = ceil(input)``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Ceil
    >>> transformer = Ceil()
    >>> transformer
    CeilTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[-0.6, -1.4, 2.2], [-1.1,  0.5, 0.2]]))
    >>> out
    tensor([[-0., -1.,  3.], [-1.,  1.,  1.]])

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
        return tensor.ceil()


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


class Expm1TensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the exponential of
    the input tensor.

    This tensor transformer is equivalent to: ``output = exp(input) - 1``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Expm1
    >>> transformer = Expm1()
    >>> transformer
    Expm1TensorTransformer()
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[  1.7183,  -0.8647,  19.0855], [ -0.9817, 147.4132,  -0.9975]])

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
        return tensor.expm1()


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


class Log1pTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the logarithm of the
    input tensor.

    This tensor transformer is equivalent to: ``output = log(input + 1)``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Log1p
    >>> transformer = Log1p()
    >>> transformer
    Log1pTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
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
        return tensor.log1p()


class PowTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes  the power of each
    element in the input tensor.

    This tensor transformer is equivalent to:
    ``output = input ** exponent``

    Args:
        exponent: The exponent value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Pow
    >>> transformer = Pow(2)
    >>> transformer
    PowTensorTransformer(exponent=2)
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[ 1.,  4.,  9.], [16., 25., 36.]])

    ```
    """

    def __init__(self, exponent: float) -> None:
        self._exponent = exponent

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(exponent={self._exponent})"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.pow(self._exponent)


class SqrtTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the square root of
    the input tensor.

    This tensor transformer is equivalent to: ``output = sqrt(input)``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Sqrt
    >>> transformer = Sqrt()
    >>> transformer
    SqrtTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[0.0, 4.0, 16.0], [1.0, 2.0, 3.0]]))
    >>> out
    tensor([[0.0000, 2.0000, 4.0000], [1.0000, 1.4142, 1.7321]])

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
        return tensor.sqrt()
