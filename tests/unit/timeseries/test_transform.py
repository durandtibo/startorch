from __future__ import annotations

import torch
from coola import objects_are_equal

from startorch.sequence import RandNormal
from startorch.tensor.transformer import Clamp
from startorch.timeseries import (
    SequenceTimeSeries,
    TransformTimeSeriesGenerator,
    VanillaTimeSeriesGenerator,
)
from startorch.transformer import TensorTransformer
from startorch.utils.seed import get_torch_generator

##################################################
#     Tests for TransformTimeSeriesGenerator     #
##################################################


def test_transform_str() -> None:
    assert str(
        TransformTimeSeriesGenerator(
            generator=VanillaTimeSeriesGenerator(
                {
                    "value": torch.arange(40, dtype=torch.float).view(4, 10),
                    "label": torch.ones(4, 10),
                }
            ),
            transformer=TensorTransformer(
                transformer=Clamp(min=2.0, max=5.0), input="value", output="value_transformed"
            ),
        )
    ).startswith("TransformTimeSeriesGenerator(")


def test_transform_generate() -> None:
    batch = TransformTimeSeriesGenerator(
        generator=VanillaTimeSeriesGenerator(
            {"value": torch.arange(40, dtype=torch.float).view(4, 10), "label": torch.ones(4, 10)}
        ),
        transformer=TensorTransformer(
            transformer=Clamp(min=2.0, max=5.0), input="value", output="value_transformed"
        ),
    ).generate(batch_size=4, seq_len=10)
    assert objects_are_equal(
        batch,
        {
            "value": torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0],
                ]
            ),
            "label": torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            ),
            "value_transformed": torch.tensor(
                [
                    [2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                ]
            ),
        },
        show_difference=True,
    )


def test_transform_generate_same_random_seed() -> None:
    generator = TransformTimeSeriesGenerator(
        generator=SequenceTimeSeries(
            {"value": RandNormal(), "time": RandNormal()},
        ),
        transformer=TensorTransformer(
            transformer=Clamp(min=-1.0, max=1.0), input="value", output="value_transformed"
        ),
    )
    assert objects_are_equal(
        generator.generate(batch_size=32, seq_len=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=32, seq_len=64, rng=get_torch_generator(1)),
    )


def test_transform_generate_different_random_seeds() -> None:
    generator = TransformTimeSeriesGenerator(
        generator=SequenceTimeSeries(
            {"value": RandNormal(), "time": RandNormal()},
        ),
        transformer=TensorTransformer(
            transformer=Clamp(min=-1.0, max=1.0), input="value", output="value_transformed"
        ),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=32, seq_len=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=32, seq_len=64, rng=get_torch_generator(2)),
    )
