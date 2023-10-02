from __future__ import annotations

__all__ = ["mix2sequences"]

from redcat import BatchedTensorSeq
from redcat.utils.common import (
    check_batch_dims,
    check_seq_dims,
    get_batch_dims,
    get_seq_dims,
)


def mix2sequences(
    x: BatchedTensorSeq, y: BatchedTensorSeq
) -> tuple[BatchedTensorSeq, BatchedTensorSeq]:
    r"""Mixes the values of two batches along the sequence dimension.

    If the input batches are
    ``x = [x[0], x[1], x[2], x[3], x[4], ...]`` and
    ``y = [y[0], y[1], y[2], y[3], y[4], ...]``, the output batches
    are: ``x = [x[0], y[1], x[2], y[3], x[4], ...]`` and
    ``y = [y[0], x[1], y[2], x[3], y[4], ...]``

    Args:
    ----
        x (``BatchedTensorSeq``): Specifies the first batch.
        y (``BatchedTensorSeq`` with the same shape as ``x``):
            Specifies the second batch.
        dim (int, optional): Specifies the dimension used to mix
            the values. Default: ``0``

    Returns:
    -------
        tuple: The batches with mixed values along the sequence
            dimension.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from redcat import BatchedTensorSeq
        >>> from startorch.timeseries.utils import mix2sequences
        >>> mix2sequences(
        ...     BatchedTensorSeq(torch.arange(10).view(2, 5)),
        ...     BatchedTensorSeq(torch.arange(10, 20).view(2, 5)),
        ... )
        (tensor([[ 0, 11,  2, 13,  4], [ 5, 16,  7, 18,  9]], batch_dim=0, seq_dim=1),
         tensor([[10,  1, 12,  3, 14], [15,  6, 17,  8, 19]], batch_dim=0, seq_dim=1))
    """
    check_batch_dims(get_batch_dims([x, y]))
    check_seq_dims(get_seq_dims([x, y]))
    if x.shape != y.shape:
        raise RuntimeError(f"x and y shapes do not match: {x.shape} vs {y.shape}")
    t = x
    seq_dim = x.seq_dim
    if seq_dim >= 2:
        x = x.align_to_seq_batch()
        y = y.align_to_seq_batch()
    z = x.clone()
    if seq_dim == 0:
        x[1::2] = y[1::2]
        y[1::2] = z[1::2]
    elif seq_dim == 1:
        x[:, 1::2] = y[:, 1::2]
        y[:, 1::2] = z[:, 1::2]
    else:
        x[1::2] = y[1::2]
        y[1::2] = z[1::2]
        x = x.align_as(t)
        y = y.align_as(t)
    return x, y
