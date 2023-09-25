from __future__ import annotations

__all__ = ["merge_timeseries_by_time"]

from collections.abc import Sequence

from redcat import BatchDict, BatchedTensorSeq

from startorch import constants as ct


def merge_timeseries_by_time(timeseries: Sequence[BatchDict], time_key: str = ct.TIME) -> BatchDict:
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
    indices = batch_time.argsort_along_seq(descending=False)
    print(indices)
    return ts.take_along_seq(indices.data)
