from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.transition import Diagonal

#################################################
#     Tests for DiagonalTransitionGenerator     #
#################################################


def test_diagonal_str() -> None:
    assert str(Diagonal()).startswith("DiagonalTransitionGenerator(")


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
def test_diagonal_generate_n_1(n: int, output: torch.Tensor) -> None:
    assert objects_are_equal(Diagonal().generate(n=n), output)
