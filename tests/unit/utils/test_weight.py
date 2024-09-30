from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

from startorch.tensor import RandNormal, RandUniform
from startorch.utils.weight import (
    GENERATOR,
    WEIGHT,
    prepare_probabilities,
    prepare_weighted_generators,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

###########################################
#     Tests for prepare_probabilities     #
###########################################


@pytest.mark.parametrize(
    ("weights", "probabilities"),
    [
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
    ],
)
def test_prepare_probabilities(weights: Tensor | Sequence, probabilities: Tensor) -> None:
    assert torch.allclose(prepare_probabilities(weights), probabilities)


def test_prepare_probabilities_incorrect_weights_dimensions() -> None:
    with pytest.raises(ValueError, match="weights has to be a 1D tensor"):
        prepare_probabilities(torch.ones(4, 5))


def test_prepare_probabilities_incorrect_weights_not_positive() -> None:
    with pytest.raises(ValueError, match="The values in weights have to be positive"):
        prepare_probabilities(torch.tensor([-1, 2, 4, 7], dtype=torch.float))


def test_prepare_probabilities_incorrect_weights_sum_zero() -> None:
    with pytest.raises(ValueError, match="The sum of the weights has to be greater than 0"):
        prepare_probabilities(torch.zeros(5))


#################################################
#     Tests for prepare_weighted_generators     #
#################################################


@pytest.mark.parametrize(
    "generators",
    [
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
        ),
        [
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
        ],
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {GENERATOR: RandNormal()},
        ),
    ],
)
def test_prepare_weighted_generators_2_generators(generators: Sequence[dict]) -> None:
    generators, weights = prepare_weighted_generators(generators)
    assert len(generators) == 2
    assert isinstance(generators[0], RandUniform)
    assert isinstance(generators[1], RandNormal)
    assert weights == (2.0, 1.0)


def test_prepare_weighted_generators_missing_weight() -> None:
    _generators, weights = prepare_weighted_generators(
        ({GENERATOR: RandUniform()}, {GENERATOR: RandNormal()})
    )
    assert weights == (1.0, 1.0)


def test_prepare_weighted_generators_empty() -> None:
    generators, weights = prepare_weighted_generators([])
    assert generators == ()
    assert weights == ()
