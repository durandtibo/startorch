from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.sequence import Full, Linear, RandUniform, VanillaSequenceGenerator
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


############################
#     Tests for Linear     #
############################


def test_linear_str() -> None:
    assert str(
        Linear(
            value=RandUniform(low=-1.0, high=1.0),
            slope=RandUniform(low=-1.0, high=1.0),
            intercept=RandUniform(low=-1.0, high=1.0),
        )
    ).startswith("LinearSequenceGenerator(")


def test_linear_generate() -> None:
    assert objects_are_equal(
        Linear(
            value=VanillaSequenceGenerator(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
            slope=VanillaSequenceGenerator(torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])),
            intercept=VanillaSequenceGenerator(torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])),
        ).generate(batch_size=2, seq_len=3),
        torch.tensor([[3.0, 5.0, 7.0], [15.0, 19.0, 23.0]]),
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_linear_generate_shape(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Linear(
        value=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        slope=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        intercept=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


def test_linear_generate_fixed() -> None:
    assert objects_are_equal(
        Linear(value=Full(2.0), slope=Full(-3.0), intercept=Full(1.0)).generate(
            batch_size=2, seq_len=4
        ),
        torch.full((2, 4, 1), -5.0),
    )


def test_linear_generate_same_random_seed() -> None:
    generator = Linear(
        value=RandUniform(low=-1.0, high=1.0),
        slope=RandUniform(low=1.0, high=2.0),
        intercept=RandUniform(low=-10.0, high=-5.0),
    )
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_linear_generate_different_random_seeds() -> None:
    generator = Linear(
        value=RandUniform(low=-1.0, high=1.0),
        slope=RandUniform(low=1.0, high=2.0),
        intercept=RandUniform(low=-10.0, high=-5.0),
    )
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
