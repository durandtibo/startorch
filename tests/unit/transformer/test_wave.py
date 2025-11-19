from __future__ import annotations

import math

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.transformer import SineWaveTransformer, sine_wave
from startorch.utils.seed import get_torch_generator

#########################################
#     Tests for SineWaveTransformer     #
#########################################


def test_sine_wave_transformer_str() -> None:
    assert str(
        SineWaveTransformer(
            value="value",
            frequency="frequency",
            phase="phase",
            amplitude="amplitude",
            output="output",
        )
    ).startswith("SineWaveTransformer(")


def test_sine_wave_transformer_transform() -> None:
    assert objects_are_allclose(
        SineWaveTransformer(
            value="value",
            frequency="frequency",
            phase="phase",
            amplitude="amplitude",
            output="output",
        ).transform(
            {
                "value": torch.tensor(
                    [
                        [0.0, 0.5, 1.0, 1.5, 2.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                    ]
                ),
                "frequency": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]]),
                "phase": torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi],
                    ]
                ),
                "amplitude": torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            }
        ),
        {
            "value": torch.tensor(
                [
                    [0.0, 0.5, 1.0, 1.5, 2.0],
                    [0.0, 0.25, 0.5, 0.75, 1.0],
                ]
            ),
            "frequency": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            "phase": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi],
                ]
            ),
            "amplitude": torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            "output": torch.tensor([[0.0, 2.0, 0.0, -2.0, 0.0], [1.0, 0.0, -1.0, 0.0, 1.0]]),
        },
        atol=1e-6,
    )


def test_sine_wave_transformer_transform_exist_ok_false() -> None:
    data = {
        "value": torch.tensor(
            [
                [0.0, 0.5, 1.0, 1.5, 2.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        ),
        "frequency": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]]),
        "phase": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi],
            ]
        ),
        "amplitude": torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
        "output": 1,
    }
    transformer = SineWaveTransformer(
        value="value",
        frequency="frequency",
        phase="phase",
        amplitude="amplitude",
        output="output",
        exist_ok=False,
    )
    with pytest.raises(KeyError, match=r"Key output already exists."):
        transformer.transform(data)


def test_sine_wave_transformer_transform_exist_ok_true() -> None:
    data = {
        "value": torch.tensor(
            [
                [0.0, 0.5, 1.0, 1.5, 2.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        ),
        "frequency": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]]),
        "phase": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi],
            ]
        ),
        "amplitude": torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
        "output": 1,
    }
    out = SineWaveTransformer(
        value="value",
        frequency="frequency",
        phase="phase",
        amplitude="amplitude",
        output="output",
        exist_ok=True,
    ).transform(data)
    assert objects_are_allclose(
        out,
        {
            "value": torch.tensor(
                [
                    [0.0, 0.5, 1.0, 1.5, 2.0],
                    [0.0, 0.25, 0.5, 0.75, 1.0],
                ]
            ),
            "frequency": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            "phase": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi],
                ]
            ),
            "amplitude": torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            "output": torch.tensor([[0.0, 2.0, 0.0, -2.0, 0.0], [1.0, 0.0, -1.0, 0.0, 1.0]]),
        },
        atol=1e-6,
    )


def test_sine_wave_transformer_transform_missing_key() -> None:
    transformer = SineWaveTransformer(
        value="value", frequency="frequency", phase="phase", amplitude="amplitude", output="output"
    )
    with pytest.raises(KeyError, match=r"Missing key: value."):
        transformer.transform({})


def test_sine_wave_transformer_transform_same_random_seed() -> None:
    transformer = SineWaveTransformer(
        value="value", frequency="frequency", phase="phase", amplitude="amplitude", output="output"
    )
    data = {
        "value": torch.randn(4, 12),
        "frequency": torch.randn(4, 12),
        "phase": torch.randn(4, 12),
        "amplitude": torch.rand(4, 12),
    }
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(1)),
    )


def test_sine_wave_transformer_transform_different_random_seeds() -> None:
    transformer = SineWaveTransformer(
        value="value", frequency="frequency", phase="phase", amplitude="amplitude", output="output"
    )
    data = {
        "value": torch.randn(4, 12),
        "frequency": torch.randn(4, 12),
        "phase": torch.randn(4, 12),
        "amplitude": torch.rand(4, 12),
    }
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(data, rng=get_torch_generator(1)),
        transformer.transform(data, rng=get_torch_generator(2)),
    )


###############################
#     Tests for sine_wave     #
###############################


def test_sine_wave() -> None:
    assert objects_are_allclose(
        sine_wave(
            value=torch.tensor(
                [
                    [0.0, 0.5, 1.0, 1.5, 2.0],
                    [0.0, 0.25, 0.5, 0.75, 1.0],
                ]
            ),
            frequency=torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            phase=torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi],
                ]
            ),
            amplitude=torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
        ),
        torch.tensor([[0.0, 2.0, 0.0, -2.0, 0.0], [1.0, 0.0, -1.0, 0.0, 1.0]]),
        atol=1e-6,
    )
