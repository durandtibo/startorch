import torch
from coola import objects_are_equal
from pytest import mark, raises

from startorch import constants as ct
from startorch.timeseries import (
    merge_multiple_timeseries_by_time,
    truncate_timeseries_by_length,
)

SIZES = (1, 2)


#######################################################
#     Tests for merge_multiple_timeseries_by_time     #
#######################################################


def test_merge_multiple_timeseries_by_time_batch_size_1() -> None:
    assert objects_are_equal(
        merge_multiple_timeseries_by_time(
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


def test_merge_multiple_timeseries_by_time_batch_size_2() -> None:
    assert objects_are_equal(
        merge_multiple_timeseries_by_time(
            [
                {
                    ct.TIME: torch.tensor(
                        [[[5], [10], [15], [20], [25]], [[4], [8], [12], [16], [20]]],
                        dtype=torch.float,
                    ),
                    ct.VALUE: torch.tensor(
                        [[111, 112, 113, 114, 115], [121, 122, 123, 124, 125]], dtype=torch.long
                    ),
                },
                {
                    ct.TIME: torch.tensor(
                        [[[6], [12], [16], [24]], [[3], [6], [9], [12]]], dtype=torch.float
                    ),
                    ct.VALUE: torch.tensor(
                        [[211, 212, 213, 214], [221, 222, 223, 224]], dtype=torch.long
                    ),
                },
            ]
        ),
        {
            ct.TIME: torch.tensor(
                [
                    [[5], [6], [10], [12], [15], [16], [20], [24], [25]],
                    [[3], [4], [6], [8], [9], [12], [12], [16], [20]],
                ],
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


def test_merge_multiple_timeseries_by_time_time_feature_1d() -> None:
    assert objects_are_equal(
        merge_multiple_timeseries_by_time(
            [
                {
                    ct.TIME: torch.tensor(
                        [[5, 10, 15, 20, 25], [4, 8, 12, 16, 20]], dtype=torch.float
                    ),
                    ct.VALUE: torch.tensor(
                        [[111, 112, 113, 114, 115], [121, 122, 123, 124, 125]], dtype=torch.long
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


def test_merge_multiple_timeseries_by_time_only_time_feature() -> None:
    assert objects_are_equal(
        merge_multiple_timeseries_by_time(
            [
                {ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float)},
                {ct.TIME: torch.tensor([[[6], [12], [16], [24]]], dtype=torch.float)},
            ]
        ),
        {
            ct.TIME: torch.tensor(
                [[[5], [6], [10], [12], [15], [16], [20], [24], [25]]], dtype=torch.float
            ),
        },
    )


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_merge_multiple_timeseries_by_time_2_inputs(batch_size: int, seq_len: int) -> None:
    output = merge_multiple_timeseries_by_time(
        [
            {
                ct.TIME: torch.rand(batch_size, seq_len, 1),
                "1d": torch.ones(batch_size, seq_len, dtype=torch.long),
                "2d": 2 * torch.ones(batch_size, seq_len, 3),
                "3d": 3 * torch.ones(batch_size, seq_len, 3, 4),
            },
            {
                ct.TIME: torch.rand(batch_size, seq_len + 1, 1),
                "1d": torch.ones(batch_size, seq_len + 1, dtype=torch.long),
                "2d": 2 * torch.ones(batch_size, seq_len + 1, 3),
                "3d": 3 * torch.ones(batch_size, seq_len + 1, 3, 4),
            },
        ]
    )
    assert len(output) == 4
    assert output[ct.TIME].shape == (batch_size, 2 * seq_len + 1, 1)
    assert output[ct.TIME].dtype == torch.float
    assert output["1d"].shape == (batch_size, 2 * seq_len + 1)
    assert output["1d"].dtype == torch.long
    assert output["2d"].shape == (batch_size, 2 * seq_len + 1, 3)
    assert output["2d"].dtype == torch.float
    assert output["3d"].shape == (batch_size, 2 * seq_len + 1, 3, 4)
    assert output["3d"].dtype == torch.float


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_merge_multiple_timeseries_by_time_3_inputs(batch_size: int, seq_len: int) -> None:
    output = merge_multiple_timeseries_by_time(
        [
            {
                ct.TIME: torch.rand(batch_size, seq_len, 1),
                "1d": torch.ones(batch_size, seq_len, dtype=torch.long),
                "2d": 2 * torch.ones(batch_size, seq_len, 3),
                "3d": 3 * torch.ones(batch_size, seq_len, 3, 4),
            },
            {
                ct.TIME: torch.rand(batch_size, seq_len + 1, 1),
                "1d": torch.ones(batch_size, seq_len + 1, dtype=torch.long),
                "2d": 2 * torch.ones(batch_size, seq_len + 1, 3),
                "3d": 3 * torch.ones(batch_size, seq_len + 1, 3, 4),
            },
            {
                ct.TIME: torch.rand(batch_size, seq_len + 2, 1),
                "1d": torch.ones(batch_size, seq_len + 2, dtype=torch.long),
                "2d": 2 * torch.ones(batch_size, seq_len + 2, 3),
                "3d": 3 * torch.ones(batch_size, seq_len + 2, 3, 4),
            },
        ]
    )
    assert len(output) == 4
    assert output[ct.TIME].shape == (batch_size, 3 * seq_len + 3, 1)
    assert output[ct.TIME].dtype == torch.float
    assert output["1d"].shape == (batch_size, 3 * seq_len + 3)
    assert output["1d"].dtype == torch.long
    assert output["2d"].shape == (batch_size, 3 * seq_len + 3, 3)
    assert output["2d"].dtype == torch.float
    assert output["3d"].shape == (batch_size, 3 * seq_len + 3, 3, 4)
    assert output["3d"].dtype == torch.float


#########################################
#     Tests for truncate_timeseries     #
#########################################


def test_truncate_timeseries_incorrect_max_seq_len() -> None:
    with raises(RuntimeError, match="`max_seq_len` has to be greater or equal to 1"):
        truncate_timeseries_by_length(
            {
                ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
                ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
            },
            max_seq_len=0,
        )


def test_truncate_timeseries_max_seq_len_1() -> None:
    assert objects_are_equal(
        truncate_timeseries_by_length(
            {
                ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
                ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
            },
            max_seq_len=1,
        ),
        {
            ct.TIME: torch.tensor([[[5]]], dtype=torch.float),
            ct.VALUE: torch.tensor([[11]], dtype=torch.long),
        },
    )


def test_truncate_timeseries_max_seq_len_3() -> None:
    assert objects_are_equal(
        truncate_timeseries_by_length(
            {
                ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
                ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
            },
            max_seq_len=3,
        ),
        {
            ct.TIME: torch.tensor([[[5], [10], [15]]], dtype=torch.float),
            ct.VALUE: torch.tensor([[11, 12, 13]], dtype=torch.long),
        },
    )


def test_truncate_timeseries_max_seq_len_5() -> None:
    assert objects_are_equal(
        truncate_timeseries_by_length(
            {
                ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
                ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
            },
            max_seq_len=5,
        ),
        {
            ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
            ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        },
    )


def test_truncate_timeseries_max_seq_len_10() -> None:
    assert objects_are_equal(
        truncate_timeseries_by_length(
            {
                ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
                ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
            },
            max_seq_len=10,
        ),
        {
            ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
            ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        },
    )
