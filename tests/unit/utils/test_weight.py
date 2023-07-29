from __future__ import annotations

from collections.abc import Sequence

import torch
from pytest import mark, raises
from torch import Tensor

from startorch.utils.weight import prepare_probabilities

###########################################
#     Tests for prepare_probabilities     #
###########################################


@mark.parametrize(
    "weights,probabilities",
    (
        (torch.ones(1), torch.ones(1)),
        (torch.ones(2), 0.5 * torch.ones(2)),
        (torch.ones(4), 0.25 * torch.ones(4)),
        (torch.ones(8), 0.125 * torch.ones(8)),
        (0.1 * torch.ones(4), 0.25 * torch.ones(4)),
        (
            torch.tensor([1, 2, 4, 7], dtype=torch.float),
            torch.tensor(
                [0.07142857142857142, 0.14285714285714285, 0.2857142857142857, 0.5],
                dtype=torch.float,
            ),
        ),
        ((1, 1, 1, 1), 0.25 * torch.ones(4)),
        ([1, 1, 1, 1], 0.25 * torch.ones(4)),
        ([2.0, 2.0, 2.0, 2.0], 0.25 * torch.ones(4)),
    ),
)
def test_prepare_probabilities(weights: Tensor | Sequence, probabilities: Tensor) -> None:
    assert torch.allclose(prepare_probabilities(weights), probabilities)


def test_prepare_probabilities_incorrect_weights_dimensions() -> None:
    with raises(ValueError, match="weights has to be a 1D tensor"):
        prepare_probabilities(torch.ones(4, 5))


def test_prepare_probabilities_incorrect_weights_not_positive() -> None:
    with raises(ValueError, match="The values in weights have to be positive"):
        prepare_probabilities(torch.tensor([-1, 2, 4, 7], dtype=torch.float))


def test_prepare_probabilities_incorrect_weights_sum_zero() -> None:
    with raises(ValueError, match="The sum of the weights has to be greater than 0"):
        prepare_probabilities(torch.zeros(5))
