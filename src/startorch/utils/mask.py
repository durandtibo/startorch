r"""Contain utility functions to mask values."""

from __future__ import annotations

__all__ = ["mask_by_row", "mask_square_matrix"]

import torch

from startorch.utils.tensor import circulant


def mask_by_row(
    tensor: torch.Tensor, n: int, mask_value: float = 0.0, rng: torch.Generator | None = None
) -> torch.Tensor:
    r"""Set to 0 some values in each row.

    Args:
        tensor: The input tensor with the data to zero.
            This input must be a 2D tensor.
        n: The number of values to mask for each row.
        mask_value: The value used to mask.
        rng: An optional random number generator.

    Returns:
        The tensor with the masked values.

    Raises:
        ValueError: if the number of dimension is not 2.
        ValueError: if number of values to mask is incorrect.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.utils.mask import mask_by_row
    >>> tensor = torch.arange(10).view(2, 5)
    >>> mask_by_row(tensor, n=2)
    tensor([[...]])

    ```
    """
    if tensor.ndim != 2:
        msg = f"Expected a 2D tensor but received a tensor with {tensor.ndim} dimensions"
        raise ValueError(msg)
    n_rows, n_cols = tensor.shape
    if n < 0 or n > n_cols:
        msg = f"Incorrect number of values to mask: {n}"
        raise ValueError(msg)
    index = torch.stack([torch.randperm(n_cols, generator=rng) for _ in range(n_rows)])[:, :n]
    tensor = tensor.clone()
    tensor.scatter_(
        dim=1,
        index=index,
        src=torch.full(size=(n_rows, n_cols), fill_value=mask_value, dtype=tensor.dtype),
    )
    return tensor


def mask_square_matrix(
    matrix: torch.Tensor, n: int, mask_value: float = 0.0, rng: torch.Generator | None = None
) -> torch.Tensor:
    r"""Set to 0 some values in each row.

    Args:
        matrix: The input tensor with the data to zero.
            This input must be a 2D tensor.
        n: The number of values to mask for each row.
        mask_value: The value used to mask.
        rng: An optional random number generator.

    Returns:
        The tensor with the masked values.

    Raises:
        ValueError: if the number of dimension is not 2.
        ValueError: if number of values to mask is incorrect.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.utils.mask import mask_by_row
    >>> tensor = torch.arange(10).view(2, 5)
    >>> mask_by_row(tensor, n=2)
    tensor([[...]])

    ```
    """
    if matrix.ndim != 2:
        msg = f"Expected a 2D tensor but received a tensor with {matrix.ndim} dimensions"
        raise ValueError(msg)
    n_rows, n_cols = matrix.shape
    if n_rows != n_cols:
        msg = f"Expected a square matrix but received {matrix.shape}"
        raise ValueError(msg)
    if n < 0 or n > n_cols:
        msg = f"Incorrect number of values to mask: {n}"
        raise ValueError(msg)
    mask = circulant(torch.arange(n_rows)) < n
    # Shuffle rows and columns
    mask = mask[torch.randperm(n_rows, generator=rng)][:, torch.randperm(n_rows, generator=rng)]
    return matrix.masked_scatter(
        mask, torch.full(size=(n_rows, n_cols), fill_value=mask_value, dtype=matrix.dtype)
    )
