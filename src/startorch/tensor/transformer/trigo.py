r"""Contain the implementation of tensor transformers that compute
trigonometric functions on tensors."""

from __future__ import annotations

__all__ = [
    "AcoshTensorTransformer",
    "AsinhTensorTransformer",
    "AtanhTensorTransformer",
    "CoshTensorTransformer",
    "SinhTensorTransformer",
    "TanhTensorTransformer",
]

from typing import TYPE_CHECKING

from startorch.tensor.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:
    import torch


class AcoshTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the inverse
    hyperbolic cosine (arccosh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Acosh
    >>> transformer = Acosh()
    >>> transformer
    AcoshTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    >>> out
    tensor([[0.0000, 1.3170, 1.7627], [2.0634, 2.2924, 2.4779]])

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
        return tensor.acosh()


class AsinhTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the inverse
    hyperbolic sine (arcsinh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Asinh
    >>> transformer = Asinh()
    >>> transformer
    AsinhTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    >>> out
    tensor([[0.8814, 1.4436, 1.8184], [2.0947, 2.3124, 2.4918]])

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
        return tensor.asinh()


class AtanhTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the inverse
    hyperbolic tangent (arctanh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Atanh
    >>> transformer = Atanh()
    >>> transformer
    AtanhTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[-0.5, -0.1, 0.0], [0.1, 0.2, 0.5]]))
    >>> out
    tensor([[-0.5493, -0.1003,  0.0000], [ 0.1003,  0.2027,  0.5493]])

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
        return tensor.atanh()


class CoshTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the hyperbolic
    cosine (cosh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Cosh
    >>> transformer = Cosh()
    >>> transformer
    CoshTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.5, 6.0]]))
    >>> out
    tensor([[  1.5431,   3.7622,  10.0677], [ 27.3082,  45.0141, 201.7156]])

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
        return tensor.cosh()


class SinhTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the hyperbolic sine
    (sinh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Sinh
    >>> transformer = Sinh()
    >>> transformer
    SinhTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]))
    >>> out
    tensor([[  0.0000,   1.1752,   3.6269], [ 27.2899,  74.2032, 201.7132]])


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
        return tensor.sinh()


class TanhTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the hyperbolic
    tangent (tanh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor.transformer import Tanh
    >>> transformer = Tanh()
    >>> transformer
    TanhTensorTransformer()
    >>> out = transformer.transform(torch.tensor([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]))
    >>> out
    tensor([[0.0000, 0.7616, 0.9640], [0.9993, 0.9999, 1.0000]])

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
        return tensor.tanh()
