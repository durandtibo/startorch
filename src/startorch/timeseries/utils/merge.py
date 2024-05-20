r"""Contain utility functions to merge batches."""

from __future__ import annotations

__all__ = ["merge_by_time"]

from typing import TYPE_CHECKING

from batchtensor.nested import cat_along_seq, index_select_along_seq
from batchtensor.tensor import argsort_along_seq

from startorch import constants as ct

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import torch


def merge_by_time(
    timeseries: Sequence[dict[Hashable, torch.Tensor]], time_key: str = ct.TIME
) -> dict[Hashable, torch.Tensor]:
    r"""Merge multiple time series by using the time.

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
        timeseries: Specifies the list of time series to merge.
            See the description above to know the format of the time
            series.
        time_key: Specifies the key that contains the time data used
            to merge the time series.

    Returns:
        The merged time series. The dictionary has the same structure
            that the input time series.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch import constants as ct
    >>> from startorch.timeseries.utils import merge_by_time
    >>> batch = merge_by_time(
    ...     [
    ...         {
    ...             ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
    ...             ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
    ...         },
    ...         {
    ...             ct.TIME: torch.tensor([[[6], [12], [16], [24]]], dtype=torch.float),
    ...             ct.VALUE: torch.tensor([[21, 22, 23, 24]], dtype=torch.long),
    ...         },
    ...     ]
    ... )
    >>> batch
    {'time': tensor([[[ 5.], [ 6.], [10.], [12.], [15.], [16.], [20.], [24.], [25.]]]),
     'value': tensor([[11, 21, 12, 22, 13, 23, 14, 24, 15]])}

    ```
    """
    if not timeseries:
        msg = "No time series is provided so it is not possible to merge time series"
        raise RuntimeError(msg)
    ts = cat_along_seq(timeseries)
    batch_time = ts.get(time_key, None)
    if batch_time is None:
        msg = f"The key {time_key} is not in batch. Available keys are: {sorted(ts.keys())}"
        raise KeyError(msg)
    indices = argsort_along_seq(batch_time, descending=False, stable=True)
    return index_select_along_seq(ts, indices)
