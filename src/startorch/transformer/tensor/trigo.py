r"""Contain the implementation of tensor transformers that computes
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

from startorch.transformer.tensor.base import BaseTensorTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class AcoshTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the inverse
    hyperbolic cosine (arccosh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer.tensor import Acosh
    >>> transformer = Acosh()
    >>> transformer
    AcoshTensorTransformer()
    >>> tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> out = transformer.transform([tensor])
    >>> out
    tensor([[0.0000, 1.3170, 1.7627],
            [2.0634, 2.2924, 2.4779]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensors: Sequence[torch.Tensor],
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        (tensor,) = tensors
        return tensor.acosh()


class AsinhTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the inverse
    hyperbolic sine (arcsinh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer.tensor import Asinh
    >>> transformer = Asinh()
    >>> transformer
    AsinhTensorTransformer()
    >>> tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> out = transformer.transform([tensor])
    >>> out
    tensor([[0.8814, 1.4436, 1.8184],
            [2.0947, 2.3124, 2.4918]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensors: Sequence[torch.Tensor],
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        (tensor,) = tensors
        return tensor.asinh()


class AtanhTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the inverse
    hyperbolic tangent (arctanh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer.tensor import Atanh
    >>> transformer = Atanh()
    >>> transformer
    AtanhTensorTransformer()
    >>> tensor = torch.tensor([[-0.5, -0.1, 0.0], [0.1, 0.2, 0.5]])
    >>> out = transformer.transform([tensor])
    >>> out
    tensor([[-0.5493, -0.1003,  0.0000],
            [ 0.1003,  0.2027,  0.5493]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensors: Sequence[torch.Tensor],
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        (tensor,) = tensors
        return tensor.atanh()


class CoshTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the hyperbolic
    cosine (cosh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer.tensor import Cosh
    >>> transformer = Cosh()
    >>> transformer
    CoshTensorTransformer()
    >>> tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> out = transformer.transform([tensor])
    >>> out
    tensor([[  1.5431,   3.7622,  10.0677],
            [ 27.3082,  74.2099, 201.7156]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensors: Sequence[torch.Tensor],
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        (tensor,) = tensors
        return tensor.cosh()


class SinhTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the hyperbolic sine
    (sinh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer.tensor import Sinh
    >>> transformer = Sinh()
    >>> transformer
    SinhTensorTransformer()
    >>> tensor = torch.tensor([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]])
    >>> out = transformer.transform([tensor])
    >>> out
    tensor([[  0.0000,   1.1752,   3.6269],
            [ 27.2899,  74.2032, 201.7132]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensors: Sequence[torch.Tensor],
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        (tensor,) = tensors
        return tensor.sinh()


class TanhTensorTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the hyperbolic
    tangent (tanh) of each value.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer.tensor import Tanh
    >>> transformer = Tanh()
    >>> transformer
    TanhTensorTransformer()
    >>> tensor = torch.tensor([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]])
    >>> out = transformer.transform([tensor])
    >>> out
    tensor([[0.0000, 0.7616, 0.9640],
            [0.9993, 0.9999, 1.0000]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        tensors: Sequence[torch.Tensor],
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        (tensor,) = tensors
        return tensor.tanh()
