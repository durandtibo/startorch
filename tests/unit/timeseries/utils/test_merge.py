import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch.timeseries.utils import merge_by_time

#############################################
#    Tests for merge_by_time     #
#############################################


def test_merge_by_time_batch_size_1() -> None:
    assert objects_are_equal(
        merge_by_time(
            [
                {
                    ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
                    ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
                },
                {
                    ct.TIME: torch.tensor([[[6], [12], [16], [24]]], dtype=torch.float),
                    ct.VALUE: torch.tensor([[21, 22, 23, 24]], dtype=torch.long),
                },
            ]
        ),
        {
            ct.TIME: torch.tensor(
                [[[5], [6], [10], [12], [15], [16], [20], [24], [25]]], dtype=torch.float
            ),
            ct.VALUE: torch.tensor([[11, 21, 12, 22, 13, 23, 14, 24, 15]], dtype=torch.long),
        },
    )


def test_merge_by_time_batch_size_2() -> None:
    assert objects_are_equal(
        merge_by_time(
            [
                {
                    ct.TIME: torch.tensor(
                        [[[5], [10], [15], [20], [25]], [[-5], [-10], [-15], [-20], [-25]]],
                        dtype=torch.float,
                    ),
                    ct.VALUE: torch.tensor(
                        [[11, 12, 13, 14, 15], [-11, -12, -13, -14, -15]], dtype=torch.long
                    ),
                },
                {
                    ct.TIME: torch.tensor(
                        [[[6], [12], [16], [24]], [[-6], [-12], [-16], [-24]]],
                        dtype=torch.float,
                    ),
                    ct.VALUE: torch.tensor(
                        [[21, 22, 23, 24], [-21, -22, -23, -24]], dtype=torch.long
                    ),
                },
            ]
        ),
        {
            ct.TIME: torch.tensor(
                [
                    [[5], [6], [10], [12], [15], [16], [20], [24], [25]],
                    [[-25], [-24], [-20], [-16], [-15], [-12], [-10], [-6], [-5]],
                ],
                dtype=torch.float,
            ),
            ct.VALUE: torch.tensor(
                [
                    [11, 21, 12, 22, 13, 23, 14, 24, 15],
                    [-15, -24, -14, -23, -13, -22, -12, -21, -11],
                ],
                dtype=torch.long,
            ),
        },
    )


def test_merge_by_time_time_feature_0d() -> None:
    assert objects_are_equal(
        merge_by_time(
            [
                {
                    ct.TIME: torch.tensor(
                        [[5, 10, 15, 20, 25], [4, 8, 12, 16, 20]], dtype=torch.float
                    ),
                    ct.VALUE: torch.tensor(
                        [[111, 112, 113, 114, 115], [121, 122, 123, 124, 125]],
                        dtype=torch.long,
                    ),
                },
                {
                    ct.TIME: torch.tensor([[6, 12, 16, 24], [3, 6, 9, 12]], dtype=torch.float),
                    ct.VALUE: torch.tensor(
                        [[211, 212, 213, 214], [221, 222, 223, 224]], dtype=torch.long
                    ),
                },
            ]
        ),
        {
            ct.TIME: torch.tensor(
                [[5, 6, 10, 12, 15, 16, 20, 24, 25], [3, 4, 6, 8, 9, 12, 12, 16, 20]],
                dtype=torch.float,
            ),
            ct.VALUE: torch.tensor(
                [
                    [111, 211, 112, 212, 113, 213, 114, 214, 115],
                    [221, 121, 222, 122, 223, 123, 224, 124, 125],
                ],
                dtype=torch.long,
            ),
        },
    )


def test_merge_by_time_only_time_feature() -> None:
    assert objects_are_equal(
        merge_by_time(
            [
                {
                    ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
                },
                {
                    ct.TIME: torch.tensor([[[6], [12], [16], [24]]], dtype=torch.float),
                },
            ]
        ),
        {
            ct.TIME: torch.tensor(
                [[[5], [6], [10], [12], [15], [16], [20], [24], [25]]], dtype=torch.float
            ),
        },
    )


def test_merge_by_time_empty() -> None:
    with pytest.raises(RuntimeError, match=r"No time series is provided so it is not possible"):
        merge_by_time([])


def test_merge_by_time_missing_time_key() -> None:
    with pytest.raises(KeyError, match=r"The key time is not in batch. Available keys are:"):
        merge_by_time(
            [
                {
                    "key": torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                    "value": torch.tensor([[100, 101, 102, 103, 104], [105, 106, 107, 108, 109]]),
                },
                {
                    "key": torch.tensor([[10, 11, 12], [13, 14, 15]]),
                    "value": torch.tensor([[110, 111, 112], [113, 114, 115]]),
                },
            ]
        )
