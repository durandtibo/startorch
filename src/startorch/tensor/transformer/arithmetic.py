r"""Contain arithmetic tensor transformers."""

from __future__ import annotations

__all__ = [
    "AddTensorTransformer",
    "DivTensorTransformer",
    "FmodTensorTransformer",
    "MulTensorTransformer",
    "NegTensorTransformer",
]

from typing import TYPE_CHECKING

from startorch.tensor.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:
    import torch


class AddTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that adds a scalar value to the
    input tensor.

    This tensor transformer is equivalent to:
    ``output = input + value``

    Args:
        value: The value to add to the input tensor.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Add
    >>> transformer = Add(1)
    >>> transformer
    AddTensorTransformer(value=1)
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[ 2., -1.,  4.], [-3.,  6., -5.]])

    ```
    """

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(value={self._value})"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.add(self._value)


class DivTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the division
    operation.

    This tensor transformer is equivalent to:
    ``output = input % divisor``

    Args:
        divisor: The divisor value.
        rounding_mode: The
            type of rounding applied to the result.
            - ``None``: true division.
            - ``"trunc"``: rounds the results of the division
                towards zero.
            - ``"floor"``: floor division.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Div
    >>> transformer = Div(divisor=4)
    >>> transformer
    DivTensorTransformer(divisor=4, rounding_mode=None)
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[ 0.2500, -0.5000,  0.7500], [-1.0000,  1.2500, -1.5000]])

    ```
    """

    def __init__(
        self,
        divisor: float,
        rounding_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._divisor = divisor
        self._rounding_mode = rounding_mode

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(divisor={self._divisor}, "
            f"rounding_mode={self._rounding_mode})"
        )

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.div(self._divisor, rounding_mode=self._rounding_mode)


class FmodTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the floating-point
    remainder of the division operation.

    This tensor transformer is equivalent to:
    ``output = input % divisor``

    Args:
        divisor: The divisor value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Fmod
    >>> transformer = Fmod(divisor=4)
    >>> transformer
    FmodTensorTransformer(divisor=4)
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[ 1., -2.,  3.], [-0.,  1., -2.]])

    ```
    """

    def __init__(self, divisor: float) -> None:
        super().__init__()
        self._divisor = divisor

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(divisor={self._divisor})"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.fmod(self._divisor)


class MulTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that multiplies a scalar value to
    the input tensor.

    This tensor transformer is equivalent to:
    ``output = input * value``

    Args:
        value: The value to multiply to the input tensor.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Mul
    >>> transformer = Mul(2)
    >>> transformer
    MulTensorTransformer(value=2)
    >>> out = transformer.transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    >>> out
    tensor([[  2.,  -4.,   6.], [ -8.,  10., -12.]])

    ```
    """

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(value={self._value})"

    def transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.mul(self._value)


class NegTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the negative of the
    input tensor.

    This tensor transformer is equivalent to: ``output = sqrt(input)``

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Neg
    >>> transformer = Neg()
    >>> transformer
    NegTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[0.0, -1.0, 2.0], [-3.0, 4.0, -5.0]]))
    >>> out
    tensor([[-0.,  1., -2.], [ 3., -4.,  5.]])

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
        return tensor.neg()
