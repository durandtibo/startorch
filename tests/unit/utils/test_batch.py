from __future__ import annotations

import torch
from pytest import raises
from redcat import BatchedTensor

from startorch.utils.batch import scale_batch

#################################
#     Tests for scale_batch     #
#################################


def test_scale_batch_identity() -> None:
    assert scale_batch(BatchedTensor(torch.arange(10).view(2, 5))).equal(
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))
    )


def test_scale_batch_log() -> None:
    assert scale_batch(BatchedTensor(torch.arange(10).add(1).view(2, 5)), scale="log").equal(
        BatchedTensor(
            torch.tensor(
                [
                    [
                        0.0,
                        0.6931471824645996,
                        1.0986123085021973,
                        1.3862943649291992,
                        1.6094379425048828,
                    ],
                    [
                        1.7917594909667969,
                        1.945910096168518,
                        2.079441547393799,
                        2.1972246170043945,
                        2.3025851249694824,
                    ],
                ]
            )
        )
    )


def test_scale_batch_log10() -> None:
    assert scale_batch(BatchedTensor(torch.arange(10).add(1).view(2, 5)), scale="log10").equal(
        BatchedTensor(
            torch.tensor(
                [
                    [
                        0.0,
                        0.3010300099849701,
                        0.4771212637424469,
                        0.6020600199699402,
                        0.6989700198173523,
                    ],
                    [
                        0.778151273727417,
                        0.8450980186462402,
                        0.9030900001525879,
                        0.9542425274848938,
                        1.0,
                    ],
                ]
            )
        )
    )


def test_scale_batch_log1p() -> None:
    assert scale_batch(BatchedTensor(torch.arange(10).view(2, 5)), scale="log1p").equal(
        BatchedTensor(
            torch.tensor(
                [
                    [
                        0.0,
                        0.6931471824645996,
                        1.0986123085021973,
                        1.3862943649291992,
                        1.6094379425048828,
                    ],
                    [
                        1.7917594909667969,
                        1.945910096168518,
                        2.079441547393799,
                        2.1972246170043945,
                        2.3025851249694824,
                    ],
                ]
            )
        )
    )


def test_scale_batch_log2() -> None:
    assert scale_batch(BatchedTensor(torch.arange(10).add(1).view(2, 5)), scale="log2").equal(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 1.0, 1.5849624872207642, 2.0, 2.321928024291992],
                    [
                        2.5849626064300537,
                        2.8073549270629883,
                        3.0,
                        3.1699249744415283,
                        3.321928024291992,
                    ],
                ]
            )
        )
    )


def test_scale_batch_asinh() -> None:
    assert scale_batch(BatchedTensor(torch.arange(10).view(2, 5)), scale="asinh").equal(
        BatchedTensor(
            torch.tensor(
                [
                    [
                        0.0,
                        0.8813735842704773,
                        1.4436354637145996,
                        1.8184465169906616,
                        2.094712495803833,
                    ],
                    [
                        2.3124382495880127,
                        2.4917798042297363,
                        2.644120693206787,
                        2.776472330093384,
                        2.893444061279297,
                    ],
                ]
            )
        )
    )


def test_scale_batch_incorrect_scale() -> None:
    with raises(RuntimeError, match="Incorrect scale: "):
        scale_batch(BatchedTensor(torch.arange(10).view(2, 5)), scale="incorrect")
