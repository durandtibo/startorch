from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.periodic.sequence import Repeat, SineWave
from startorch.sequence import Arange, Full, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


##############################
#     Tests for SineWave     #
##############################


def test_sine_wave_str() -> None:
    assert str(
        SineWave(
            value=Repeat(RandUniform(low=-1.0, high=1.0)),
            phase=Repeat(RandUniform(low=-1.0, high=1.0)),
            amplitude=Repeat(RandUniform(low=-1.0, high=1.0)),
        )
    ).startswith("SineWavePeriodicSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("period", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_sine_wave_generate(batch_size: int, seq_len: int, period: int, feature_size: int) -> None:
    batch = SineWave(
        value=Repeat(RandUniform(low=-1.0, high=1.0, feature_size=feature_size)),
        phase=Repeat(RandUniform(low=-1.0, high=1.0, feature_size=feature_size)),
        amplitude=Repeat(RandUniform(low=-1.0, high=1.0, feature_size=feature_size)),
    ).generate(batch_size=batch_size, period=period, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_sine_wave_generate_fixed() -> None:
    assert objects_are_equal(
        SineWave(
            value=Repeat(Arange()),
            phase=Repeat(Full(0.0)),
            amplitude=Repeat(Full(1.0)),
        ).generate(batch_size=1, period=1, seq_len=4),
        torch.zeros(1, 4, 1),
    )


def test_sine_wave_generate_same_random_seed() -> None:
    generator = SineWave(
        value=Repeat(RandUniform(low=-1.0, high=1.0)),
        phase=Repeat(RandUniform(low=-1.0, high=1.0)),
        amplitude=Repeat(RandUniform(low=-1.0, high=1.0)),
    )
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)),
    )


def test_sine_wave_generate_different_random_seeds() -> None:
    generator = SineWave(
        value=Repeat(RandUniform(low=-1.0, high=1.0)),
        phase=Repeat(RandUniform(low=-1.0, high=1.0)),
        amplitude=Repeat(RandUniform(low=-1.0, high=1.0)),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(2)),
    )
