from __future__ import annotations

import math

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.sequence import (
    Cumsum,
    Full,
    RandUniform,
    SineWave,
    VanillaSequenceGenerator,
)
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


##############################
#     Tests for SineWave     #
##############################


def test_sine_wave_str() -> None:
    assert str(
        SineWave(
            value=RandUniform(low=-1.0, high=1.0),
            frequency=RandUniform(low=-1.0, high=1.0),
            phase=RandUniform(low=-1.0, high=1.0),
            amplitude=RandUniform(low=-1.0, high=1.0),
        )
    ).startswith("SineWaveSequenceGenerator(")


def test_sine_wave_generate() -> None:
    assert objects_are_allclose(
        SineWave(
            value=VanillaSequenceGenerator(
                torch.tensor(
                    [
                        [0.0, 0.5, 1.0, 1.5, 2.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                    ]
                )
            ),
            frequency=VanillaSequenceGenerator(
                torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]])
            ),
            phase=VanillaSequenceGenerator(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi],
                    ]
                )
            ),
            amplitude=VanillaSequenceGenerator(
                torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
            ),
        ).generate(batch_size=2, seq_len=5),
        torch.tensor([[0.0, 2.0, 0.0, -2.0, 0.0], [1.0, 0.0, -1.0, 0.0, 1.0]]),
        atol=1e-6,
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_sine_wave_generate_shape(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = SineWave(
        value=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        frequency=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        phase=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        amplitude=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


def test_sine_wave_generate_fixed() -> None:
    assert objects_are_allclose(
        SineWave(
            value=Cumsum(Full(1.0)),
            frequency=Full(1.0),
            phase=Full(0.0),
            amplitude=Full(1.0),
        ).generate(batch_size=1, seq_len=4),
        torch.arange(1, 5).mul(2 * math.pi).sin().view(1, 4, 1),
    )


def test_sine_wave_generate_same_random_seed() -> None:
    generator = SineWave(
        value=RandUniform(low=-1.0, high=1.0),
        frequency=RandUniform(low=-1.0, high=1.0),
        phase=RandUniform(low=-1.0, high=1.0),
        amplitude=RandUniform(low=-1.0, high=1.0),
    )
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_sine_wave_generate_different_random_seeds() -> None:
    generator = SineWave(
        value=RandUniform(low=-1.0, high=1.0),
        frequency=RandUniform(low=-1.0, high=1.0),
        phase=RandUniform(low=-1.0, high=1.0),
        amplitude=RandUniform(low=-1.0, high=1.0),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
