import torch
from coola import objects_are_equal
from pytest import raises
from redcat import BatchDict, BatchedTensorSeq, BatchList

from startorch import constants as ct
from startorch.timeseries.utils.merge import merge_timeseries_by_time

#############################################
#    Tests for merge_timeseries_by_time     #
#############################################


def test_merge_timeseries_by_time_batch_size_1() -> None:
    assert objects_are_equal(
        merge_timeseries_by_time(
            [
                BatchDict(
                    {
                        ct.TIME: BatchedTensorSeq(
                            torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float)
                        ),
                        ct.VALUE: BatchedTensorSeq(
                            torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long)
                        ),
                    }
                ),
                BatchDict(
                    {
                        ct.TIME: BatchedTensorSeq(
                            torch.tensor([[[6], [12], [16], [24]]], dtype=torch.float)
                        ),
                        ct.VALUE: BatchedTensorSeq(
                            torch.tensor([[21, 22, 23, 24]], dtype=torch.long)
                        ),
                    }
                ),
            ]
        ),
        BatchDict(
            {
                ct.TIME: BatchedTensorSeq(
                    torch.tensor(
                        [[[5], [6], [10], [12], [15], [16], [20], [24], [25]]], dtype=torch.float
                    )
                ),
                ct.VALUE: BatchedTensorSeq(
                    torch.tensor([[11, 21, 12, 22, 13, 23, 14, 24, 15]], dtype=torch.long)
                ),
            }
        ),
        show_difference=True,
    )


def test_merge_timeseries_by_time_empty() -> None:
    with raises(RuntimeError, match="No time series is provided so it is not possible"):
        merge_timeseries_by_time([])


def test_merge_timeseries_by_time_missing_time_key() -> None:
    with raises(KeyError, match="The key time is not in batch. Available keys are:"):
        merge_timeseries_by_time(
            [
                BatchDict(
                    {
                        "key": BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])),
                        "value": BatchedTensorSeq(
                            torch.tensor([[100, 101, 102, 103, 104], [105, 106, 107, 108, 109]])
                        ),
                    }
                ),
                BatchDict(
                    {
                        "key": BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
                        "value": BatchedTensorSeq(torch.tensor([[110, 111, 112], [113, 114, 115]])),
                    }
                ),
            ]
        )


def test_merge_timeseries_by_time_invalid_time() -> None:
    with raises(TypeError, match="Invalid time batch type. Expected BatchedTensorSeq"):
        merge_timeseries_by_time([BatchDict({ct.TIME: BatchList(["a", "b"])})])
