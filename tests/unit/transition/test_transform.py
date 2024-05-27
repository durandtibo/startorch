from __future__ import annotations

import torch
from coola import objects_are_equal

from startorch.tensor.transformer import Clamp
from startorch.transition import Diagonal, Transform
from startorch.utils.seed import get_torch_generator

##################################################
#     Tests for TransformTransitionGenerator     #
##################################################


def test_transform_str() -> None:
    assert str(Transform(Diagonal(), Clamp(min=0.0, max=0.5))).startswith(
        "TransformTransitionGenerator("
    )


def test_transform_generate() -> None:
    assert objects_are_equal(
        Transform(Diagonal(), Clamp(min=0.0, max=0.5)).generate(n=6),
        torch.tensor(
            [
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            ]
        ),
    )


def test_transform_generate_same_random_seed() -> None:
    generator = Transform(Diagonal(), Clamp(min=0.0, max=0.5))
    assert objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(1)),
    )


def test_transform_generate_different_random_seeds() -> None:
    generator = Transform(Diagonal(), Clamp(min=0.0, max=0.5))
    # they are equal because the generator does not have randomness
    assert objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(2)),
    )
