from __future__ import annotations

__all__ = ["scale_batch"]

from redcat import BatchedTensor


def scale_batch(batch: BatchedTensor, scale: str = "identity") -> BatchedTensor:
    r"""Scales a batch.

    Args:
    ----
        batch (``BatchedTensor``): Specifies the batch to scale.
        scale (str, optional): Specifies the scaling transformation.
            Default: ``'identity'``

    Returns:
    -------
        ``BatchedTensor``: The scaled batch.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from redcat import BatchedTensor
        >>> from startorch.utils.batch import scale_batch
        >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
        >>> scale_batch(batch, scale="asinh")
        tensor([[0.0000, 0.8814, 1.4436, 1.8184, 2.0947],
                [2.3124, 2.4918, 2.6441, 2.7765, 2.8934]], batch_dim=0)
    """
    valid = {"identity", "log", "log10", "log2", "log1p", "asinh"}
    if scale not in valid:
        raise RuntimeError(f"Incorrect scale: {scale}. Valid scales are: {valid}")
    if scale == "identity":
        return batch
    return getattr(batch, scale)()
