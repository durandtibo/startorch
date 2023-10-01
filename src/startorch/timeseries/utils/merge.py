from __future__ import annotations

__all__ = ["merge_timeseries_by_time"]

from collections.abc import Sequence

from redcat import BatchDict, BatchedTensorSeq

from startorch import constants as ct


def merge_timeseries_by_time(
    timeseries: Sequence[BatchDict[BatchedTensorSeq]], time_key: str = ct.TIME
) -> BatchDict[BatchedTensorSeq]:
    r"""Merges multiple time series by using the time.

    The sequence are merged based on the time information i.e. the
    merged time series is sorted by time. The input is a list of
    time series. Each time series is represented by a dictionary of
    ``BatchedTensorSeq``s. All the values in the dictionary have to be
    of type ``BatchedTensorSeq``. Each time series has to have a time
    tensor, and this time tensor is used to sort the merged time
    series. This function only supports the time tensor where
    the time is encoded as a single scalar (float, int, or long).
    The shape of the time batch should be
    ``(batch_size, sequence_length, 1)`` or
    ``(batch_size, sequence_length)``. For example, the time in
    second can be used to encode the time information. All the
    batches in each dictionary should have a shape
    ``(batch_size, sequence_length, *)`` where `*` means any
    number of dimensions. Only the sequence dimension can change
    between the time series. This function works for a variable
    number of features.

    Args:
    ----
        timeseries (``Sequence``): Specifies the list of time series
            to merge. See the description above to know the format of
            the time series.
        time_key (str, optional): Specifies the key that contains the
            time information used to merge the time series.
            Default: ``'time'``

    Returns:
    -------
        ``BatchDict``: The merged time series. The dictionary has the
            same structure that the input time series.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch import constants as ct
        >>> from startorch.timeseries.utils import merge_timeseries_by_time
        >>> batch = merge_timeseries_by_time(
        ...     [
        ...         BatchDict(
        ...             {
        ...                 ct.TIME: BatchedTensorSeq(
        ...                     torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float)
        ...                 ),
        ...                 ct.VALUE: BatchedTensorSeq(
        ...                     torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
        ...                 ),
        ...             }
        ...         ),
        ...         BatchDict(
        ...             {
        ...                 ct.TIME: BatchedTensorSeq(
        ...                     torch.tensor([[[6], [12], [16], [24]]], dtype=torch.float)
        ...                 ),
        ...                 ct.VALUE: BatchedTensorSeq(
        ...                     torch.tensor([[21, 22, 23, 24]], dtype=torch.long)
        ...                 ),
        ...             }
        ...         ),
        ...     ]
        ... )
        >>> batch
        BatchDict(
          (time): tensor([[[ 5.], [ 6.], [10.], [12.], [15.], [16.], [20.], [24.], [25.]]], batch_dim=0, seq_dim=1)
          (value): tensor([[11, 21, 12, 22, 13, 23, 14, 24, 15]], batch_dim=0, seq_dim=1)
        )
    """
    if not timeseries:
        raise RuntimeError("No time series is provided so it is not possible to merge time series")
    ts = timeseries[0].cat_along_seq(timeseries[1:])
    batch_time = ts.get(time_key, None)
    if batch_time is None:
        raise KeyError(
            f"The key {time_key} is not in batch. Available keys are: {sorted(ts.data.keys())}"
        )
    if not isinstance(batch_time, BatchedTensorSeq):
        raise TypeError(
            f"Invalid time batch type. Expected BatchedTensorSeq but received: {type(batch_time)}"
        )
    indices = batch_time.argsort_along_seq(descending=False, stable=True)
    return ts.index_select_along_seq(indices.data)
