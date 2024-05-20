r"""Contain the implementation of tensor transformers that computes
trigonometric functions on tensors."""

from __future__ import annotations

__all__ = [
    "AcoshTransformer",
    "AsinhTransformer",
    "AtanhTransformer",
    "CoshTransformer",
    "SinhTransformer",
    "TanhTransformer",
]

from typing import TYPE_CHECKING

from startorch.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:
    import torch


class AcoshTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the inverse
    hyperbolic cosine (arccosh) of each value.

    Args:
        input: The key that contains the input tensor.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Acosh
    >>> transformer = Acosh(input="input", output="output")
    >>> transformer
    AcoshTransformer(input=input, output=output, exist_ok=False)
    >>> data = {"input": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'input': tensor([[1., 2., 3.],
                      [4., 5., 6.]]),
     'output': tensor([[0.0000, 1.3170, 1.7627],
                       [2.0634, 2.2924, 2.4779]])}

    ```
    """

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.acosh()


class AsinhTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the inverse
    hyperbolic sine (arcsinh) of each value.

    Args:
        input: The key that contains the input tensor.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Asinh
    >>> transformer = Asinh(input="input", output="output")
    >>> transformer
    AsinhTransformer(input=input, output=output, exist_ok=False)
    >>> data = {'input': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'input': tensor([[1., 2., 3.],
                      [4., 5., 6.]]),
     'output': tensor([[0.8814, 1.4436, 1.8184],
                       [2.0947, 2.3124, 2.4918]])}

    ```
    """

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.asinh()


class AtanhTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the inverse
    hyperbolic tangent (arctanh) of each value.

    Args:
        input: The key that contains the input tensor.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Atanh
    >>> transformer = Atanh(input="input", output="output")
    >>> transformer
    AtanhTransformer(input=input, output=output, exist_ok=False)
    >>> data = {'input': torch.tensor([[-0.5, -0.1, 0.0], [0.1, 0.2, 0.5]])}
    >>> out = transformer.transform(data)
    >>> out
    {'input': tensor([[-0.5000, -0.1000,  0.0000],
                      [ 0.1000,  0.2000,  0.5000]]),
     'output': tensor([[-0.5493, -0.1003,  0.0000],
                       [ 0.1003,  0.2027,  0.5493]])}

    ```
    """

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.atanh()


class CoshTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the hyperbolic
    cosine (cosh) of each value.

    Args:
        input: The key that contains the input tensor.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Cosh
    >>> transformer = Cosh(input="input", output="output")
    >>> transformer
    CoshTransformer(input=input, output=output, exist_ok=False)
    >>> data = {'input': torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.5, 6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'input': tensor([[1.0000, 2.0000, 3.0000],
                      [4.0000, 4.5000, 6.0000]]),
     'output': tensor([[  1.5431,   3.7622,  10.0677],
                       [ 27.3082,  45.0141, 201.7156]])}

    ```
    """

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.cosh()


class SinhTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the hyperbolic sine
    (sinh) of each value.

    Args:
        input: The key that contains the input tensor.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Sinh
    >>> transformer = Sinh(input="input", output="output")
    >>> transformer
    SinhTransformer(input=input, output=output, exist_ok=False)
    >>> data = {'input': torch.tensor([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'input': tensor([[0., 1., 2.],
                      [4., 5., 6.]]),
     'output': tensor([[  0.0000,   1.1752,   3.6269],
                       [ 27.2899,  74.2032, 201.7132]])}


    ```
    """

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.sinh()


class TanhTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that computes the hyperbolic
    tangent (tanh) of each value.

    Args:
        input: The key that contains the input tensor.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Tanh
    >>> transformer = Tanh(input="input", output="output")
    >>> transformer
    TanhTransformer(input=input, output=output, exist_ok=False)
    >>> data = {"input": torch.tensor([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'input': tensor([[0., 1., 2.],
                      [4., 5., 6.]]),
     'output': tensor([[0.0000, 0.7616, 0.9640],
                       [0.9993, 0.9999, 1.0000]])}

    ```
    """

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return tensor.tanh()
