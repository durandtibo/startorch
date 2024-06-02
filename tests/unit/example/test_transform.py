from __future__ import annotations

import torch
from coola import objects_are_equal

from startorch.example import (
    HypercubeClassification,
    TransformExampleGenerator,
    VanillaExampleGenerator,
)
from startorch.tensor.transformer import Clamp
from startorch.transformer import TensorTransformer
from startorch.utils.seed import get_torch_generator

###############################################
#     Tests for TransformExampleGenerator     #
###############################################


def test_transform_str() -> None:
    assert str(
        TransformExampleGenerator(
            generator=VanillaExampleGenerator(
                {"value": torch.arange(30).view(10, 3), "label": torch.arange(10)}
            ),
            transformer=TensorTransformer(
                transformer=Clamp(min=2.0, max=5.0), input="value", output="value_transformed"
            ),
        )
    ).startswith("TransformExampleGenerator(")


def test_transform_generate() -> None:
    batch = TransformExampleGenerator(
        generator=VanillaExampleGenerator(
            {"value": torch.arange(30, dtype=torch.float).view(10, 3), "label": torch.arange(10)}
        ),
        transformer=TensorTransformer(
            transformer=Clamp(min=2.0, max=5.0), input="value", output="value_transformed"
        ),
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


def test_transform_generate_same_random_seed() -> None:
    generator = TransformExampleGenerator(
        generator=HypercubeClassification(num_classes=5, feature_size=6),
        transformer=TensorTransformer(
            transformer=Clamp(min=-1.0, max=1.0), input="feature", output="feature_transformed"
        ),
    )
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


def test_transform_generate_different_random_seeds() -> None:
    generator = TransformExampleGenerator(
        generator=HypercubeClassification(num_classes=5, feature_size=6),
        transformer=TensorTransformer(
            transformer=Clamp(min=-1.0, max=1.0), input="feature", output="vfeature_transformed"
        ),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )
