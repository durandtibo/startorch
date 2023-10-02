from __future__ import annotations

import torch
from coola import objects_are_equal
from pytest import raises
from redcat import BatchedTensorSeq

from startorch.timeseries.utils import mix2sequences

###################################
#     Tests for mix2sequences     #
###################################


def test_mix2sequences_seq_dim_0() -> None:
    assert objects_are_equal(
        mix2sequences(
            BatchedTensorSeq.from_seq_batch(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])),
            BatchedTensorSeq.from_seq_batch(
                torch.tensor([[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]])
            ),
        ),
        (
            BatchedTensorSeq.from_seq_batch(
                torch.tensor([[0, 1], [12, 13], [4, 5], [16, 17], [8, 9]])
            ),
            BatchedTensorSeq.from_seq_batch(
                torch.tensor([[10, 11], [2, 3], [14, 15], [6, 7], [18, 19]])
            ),
        ),
    )


def test_mix2sequences_seq_dim_1() -> None:
    assert objects_are_equal(
        mix2sequences(
            BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]])),
            BatchedTensorSeq(torch.tensor([[10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5]])),
        ),
        (
            BatchedTensorSeq(torch.tensor([[0, 11, 2, 13, 4, 15], [10, 1, 12, 3, 14, 5]])),
            BatchedTensorSeq(torch.tensor([[10, 1, 12, 3, 14, 5], [0, 11, 2, 13, 4, 15]])),
        ),
    )


def test_mix2sequences_seq_dim_2() -> None:
    assert objects_are_equal(
        mix2sequences(
            BatchedTensorSeq(torch.arange(10).view(1, 2, 5), seq_dim=2),
            BatchedTensorSeq(torch.arange(10, 20).view(1, 2, 5), seq_dim=2),
        ),
        (
            BatchedTensorSeq(torch.tensor([[[0, 11, 2, 13, 4], [5, 16, 7, 18, 9]]]), seq_dim=2),
            BatchedTensorSeq(torch.tensor([[[10, 1, 12, 3, 14], [15, 6, 17, 8, 19]]]), seq_dim=2),
        ),
    )


def test_mix2sequences_incorrect_shape() -> None:
    with raises(RuntimeError, match="x and y shapes do not match:"):
        mix2sequences(BatchedTensorSeq(torch.ones(1, 5)), BatchedTensorSeq(torch.ones(1, 5, 1)))
