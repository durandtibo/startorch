from __future__ import annotations

import torch
from coola import objects_are_equal

from startorch.example import (
    HypercubeClassification,
    Transformed,
    VanillaExampleGenerator,
)
from startorch.transformer import Clamp
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


#################################################
#     Tests for TransformedExampleGenerator     #
#################################################


def test_generator_str() -> None:
    assert str(
        Transformed(
            generator=VanillaExampleGenerator(
                {"value": torch.arange(30).view(10, 3), "label": torch.arange(10)}
            ),
            transformer=Clamp(input="value", output="value_transformed", min=2.0, max=5.0),
        )
    ).startswith("TransformedExampleGenerator(")


def test_generator_generate() -> None:
    batch = Transformed(
        generator=VanillaExampleGenerator(
            {"value": torch.arange(30, dtype=torch.float).view(10, 3), "label": torch.arange(10)}
        ),
        transformer=Clamp(input="value", output="value_transformed", min=2.0, max=5.0),
    ).generate(batch_size=5)
    assert objects_are_equal(
        batch,
        {
            "value": torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                ],
                dtype=torch.float,
            ),
            "value_transformed": torch.tensor(
                [
                    [2.0, 2.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0],
                ],
                dtype=torch.float,
            ),
            "label": torch.tensor([0, 1, 2, 3, 4]),
        },
    )


def test_generator_generate_same_random_seed() -> None:
    generator = Transformed(
        generator=HypercubeClassification(num_classes=5, feature_size=6),
        transformer=Clamp(input="feature", output="feature_transformed", min=-1.0, max=1.0),
    )
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


def test_generator_generate_different_random_seeds() -> None:
    generator = Transformed(
        generator=HypercubeClassification(num_classes=5, feature_size=6),
        transformer=Clamp(input="feature", output="feature_transformed", min=-1.0, max=1.0),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )
