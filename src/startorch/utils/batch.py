r"""Contain utility functions for batches."""

from __future__ import annotations

__all__ = ["merge_batches", "scale_batch"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from redcat import BatchedTensor


def merge_batches(batches: Sequence[BatchedTensor]) -> BatchedTensor:
    r"""Merge batches into a single batch.

    Args:
        batches: Specifies the batches to merge.

    Returns:
        The "merged" batch.

    Example usage:

    ```pycon
    >>> import torch
    >>> from redcat import BatchedTensor
    >>> from startorch.utils.batch import merge_batches
    >>> batch = merge_batches(
    ...     [
    ...         BatchedTensor(torch.arange(6).view(2, 3)),
    ...         BatchedTensor(torch.ones(2, 3)),
    ...         BatchedTensor(torch.zeros(1, 3)),
    ...     ]
    ... )
    >>> batch
    tensor([[0., 1., 2.],
            [3., 4., 5.],
            [1., 1., 1.],
            [1., 1., 1.],
            [0., 0., 0.]], batch_dim=0)

    ```
    """
    if not batches:
        msg = "No batch is provided"
        raise RuntimeError(msg)
    batch = batches[0].clone()
    batch.extend(batches[1:])
    return batch


def scale_batch(batch: BatchedTensor, scale: str = "identity") -> BatchedTensor:
    r"""Scales a batch.

    Args:
        batch: Specifies the batch to scale.
        scale: Specifies the scaling transformation.

    Returns:
        The scaled batch.

    Example usage:

    ```pycon
    >>> import torch
    >>> from redcat import BatchedTensor
    >>> from startorch.utils.batch import scale_batch
    >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
    >>> scale_batch(batch, scale="asinh")
    tensor([[0.0000, 0.8814, 1.4436, 1.8184, 2.0947],
            [2.3124, 2.4918, 2.6441, 2.7765, 2.8934]], batch_dim=0)

    ```
    """
    valid = {"identity", "log", "log10", "log2", "log1p", "asinh"}
    if scale not in valid:
        msg = f"Incorrect scale: {scale}. Valid scales are: {valid}"
        raise RuntimeError(msg)
    if scale == "identity":
        return batch
    return getattr(batch, scale)()
