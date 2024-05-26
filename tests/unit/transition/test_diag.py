from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.transition import Diagonal, PermutedDiagonal
from startorch.utils.seed import get_torch_generator

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
def test_diagonal_generate(n: int, output: torch.Tensor) -> None:
    assert objects_are_equal(Diagonal().generate(n=n), output)


def test_diagonal_generate_same_random_seed() -> None:
    generator = Diagonal()
    assert objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(1)),
    )


def test_diagonal_generate_different_random_seeds() -> None:
    generator = Diagonal()
    # they are equal because the generator does not have randomness
    assert objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(2)),
    )


#########################################################
#     Tests for PermutedDiagonalTransitionGenerator     #
#########################################################


def test_permuted_diagonal_str() -> None:
    assert str(PermutedDiagonal()).startswith("PermutedDiagonalTransitionGenerator(")


@pytest.mark.parametrize("n", [1, 2, 6])
def test_permuted_diagonal_generate(n: int) -> None:
    out = PermutedDiagonal().generate(n=n)
    assert out.shape == (n, n)
    assert out.dtype == torch.float
    assert objects_are_equal(out.sum(dim=0), torch.ones(n))
    assert objects_are_equal(out.sum(dim=1), torch.ones(n))


def test_permuted_diagonal_generate_same_random_seed() -> None:
    generator = PermutedDiagonal()
    assert objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(1)),
    )


def test_permuted_diagonal_generate_different_random_seeds() -> None:
    generator = PermutedDiagonal()
    assert not objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(2)),
    )
