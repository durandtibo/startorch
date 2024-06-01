from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.sequence import Full, RandNormal, TransformSequenceGenerator
from startorch.tensor.transformer import Abs
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


################################################
#     Tests for TransformSequenceGenerator     #
################################################


def test_transform_str() -> None:
    assert str(TransformSequenceGenerator(generator=RandNormal(), transformer=Abs())).startswith(
        "TransformSequenceGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_transform_generate(batch_size: int, seq_len: int) -> None:
    generator = TransformSequenceGenerator(generator=RandNormal(), transformer=Abs())
    tensor = generator.generate(batch_size=batch_size, seq_len=seq_len)
    assert tensor.shape == (batch_size, seq_len, 1)
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0


def test_transform_generate_fixed_value() -> None:
    assert (
        TransformSequenceGenerator(generator=Full(-1), transformer=Abs())
        .generate(batch_size=2, seq_len=4)
        .equal(torch.ones(2, 4, 1))
    )


def test_transform_generate_same_random_seed() -> None:
    generator = TransformSequenceGenerator(generator=RandNormal(), transformer=Abs())
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_transform_generate_different_random_seeds() -> None:
    generator = TransformSequenceGenerator(generator=RandNormal(), transformer=Abs())
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
