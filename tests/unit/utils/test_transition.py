from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.utils.transition import DiagonalTransitionGenerator

#################################################
#     Tests for DiagonalTransitionGenerator     #
#################################################


def test_diagonal_transition_generator_str() -> None:
    assert str(DiagonalTransitionGenerator()).startswith("DiagonalTransitionGenerator(")


@pytest.mark.parametrize(
    ("n", "output"),
    [
        (1, torch.tensor([[1.0]])),
        (2, torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
        (
            6,
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
    ],
)
def test_diagonal_transition_generator_generate_n_1(n: int, output: torch.Tensor) -> None:
    assert objects_are_equal(DiagonalTransitionGenerator().generate(n=n), output)
