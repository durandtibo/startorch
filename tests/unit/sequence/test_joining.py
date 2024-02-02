from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.sequence import Cat2, RandUniform
from startorch.tensor import RandInt
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


##########################
#     Tests for Cat2     #
##########################


def test_cat2_str() -> None:
    assert str(
        Cat2(
            generator1=RandUniform(low=-1.0, high=0.0),
            generator2=RandUniform(low=0.0, high=1.0),
            changepoint=RandInt(low=5, high=10),
        )
    ).startswith("Cat2SequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_cat2_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Cat2(
        generator1=RandUniform(low=-1.0, high=0.0, feature_size=feature_size),
        generator2=RandUniform(low=0.0, high=1.0, feature_size=feature_size),
        changepoint=RandInt(low=5, high=10),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


def test_cat2_generate_negative_changepoint() -> None:
    batch = Cat2(
        generator1=RandUniform(low=-2.0, high=-1.0),
        generator2=RandUniform(low=1.0, high=2.0),
        changepoint=RandInt(low=-10, high=-5),
    ).generate(batch_size=4, seq_len=12)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 12, 1)
    assert batch.min() > 0.0


def test_cat2_generate_large_changepoint() -> None:
    batch = Cat2(
        generator1=RandUniform(low=-2.0, high=-1.0),
        generator2=RandUniform(low=1.0, high=2.0),
        changepoint=RandInt(low=100, high=200),
    ).generate(batch_size=4, seq_len=12)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 12, 1)
    assert batch.max() < 0.0


def test_cat2_generate_same_random_seed() -> None:
    generator = Cat2(
        generator1=RandUniform(low=-1.0, high=0.0),
        generator2=RandUniform(low=0.0, high=1.0),
        changepoint=RandInt(low=5, high=10),
    )
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_cat2_generate_different_random_seeds() -> None:
    generator = Cat2(
        generator1=RandUniform(low=-1.0, high=0.0),
        generator2=RandUniform(low=0.0, high=1.0),
        changepoint=RandInt(low=5, high=10),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
