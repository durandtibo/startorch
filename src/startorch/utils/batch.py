r"""Contain utility functions for batches."""

from __future__ import annotations

__all__ = ["scale_batch"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def scale_batch(batch: torch.Tensor, scale: str = "identity") -> torch.Tensor:
    r"""Scale a batch.

    Args:
        batch: Specifies the batch to scale.
        scale: Specifies the scaling transformation.

    Returns:
        The scaled batch.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.utils.batch import scale_batch
    >>> batch = torch.arange(10).view(2, 5)
    >>> scale_batch(batch, scale="asinh")
    tensor([[0.0000, 0.8814, 1.4436, 1.8184, 2.0947],
            [2.3124, 2.4918, 2.6441, 2.7765, 2.8934]])

    ```
    """
    valid = {"identity", "log", "log10", "log2", "log1p", "asinh"}
    if scale not in valid:
        msg = f"Incorrect scale: {scale}. Valid scales are: {valid}"
        raise RuntimeError(msg)
    if scale == "identity":
        return batch
    return getattr(batch, scale)()
