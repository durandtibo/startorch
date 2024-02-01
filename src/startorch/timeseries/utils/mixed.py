r"""Contain utility functions to mix batches."""

from __future__ import annotations

__all__ = ["mix2sequences"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def mix2sequences(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Mix the values of two batches along the sequence dimension.

    If the input batches are
    ``x = [x[0], x[1], x[2], x[3], x[4], ...]`` and
    ``y = [y[0], y[1], y[2], y[3], y[4], ...]``, the output batches
    are: ``x = [x[0], y[1], x[2], y[3], x[4], ...]`` and
    ``y = [y[0], x[1], y[2], x[3], y[4], ...]``

    Args:
        x: Specifies the first batch.
        y: Specifies the second batch. It must have the same shape
            as ``x``.

    Returns:
        The batches with mixed values along the sequence dimension.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.timeseries.utils import mix2sequences
    >>> mix2sequences(
    ...     torch.arange(10).view(2, 5),
    ...     torch.arange(10, 20).view(2, 5),
    ... )
    (tensor([[ 0, 11,  2, 13,  4], [ 5, 16,  7, 18,  9]]),
     tensor([[10,  1, 12,  3, 14], [15,  6, 17,  8, 19]]))

    ```
    """
    if x.shape != y.shape:
        msg = f"x and y shapes do not match: {x.shape} vs {y.shape}"
        raise RuntimeError(msg)
    z = x.clone()
    x[:, 1::2] = y[:, 1::2]
    y[:, 1::2] = z[:, 1::2]
    return x, y
