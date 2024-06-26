from __future__ import annotations

import pytest
import torch
from batchtensor.tensor import select_along_seq
from coola import objects_are_equal

from startorch.sequence import Constant, Full, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


##############################
#     Tests for Constant     #
##############################


def test_constant_str() -> None:
    assert str(Constant(RandUniform())).startswith("ConstantSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_constant_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Constant(RandUniform(feature_size=feature_size)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


def test_constant_generate_constant() -> None:
    batch = Constant(RandUniform(feature_size=3)).generate(batch_size=6, seq_len=4)
    assert objects_are_equal(select_along_seq(batch, 0), select_along_seq(batch, 1))
    assert objects_are_equal(select_along_seq(batch, 0), select_along_seq(batch, 2))
    assert objects_are_equal(select_along_seq(batch, 0), select_along_seq(batch, 3))


def test_constant_generate_same_random_seed() -> None:
    generator = Constant(RandUniform())
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_constant_generate_different_random_seeds() -> None:
    generator = Constant(RandUniform())
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )


##########################
#     Tests for Full     #
##########################


def test_full_str() -> None:
    assert str(Full(value=42.0)).startswith("FullSequenceGenerator(")


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_full_value(value: float) -> None:
    assert Full(value=value)._value == value


def test_full_feature_size_default() -> None:
    assert Full(value=42.0)._feature_size == (1,)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_full_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    assert objects_are_equal(
        Full(value=42.0).generate(batch_size=batch_size, seq_len=seq_len),
        torch.full((batch_size, seq_len, 1), 42.0),
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_full_generate_feature_size_int(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert objects_are_equal(
        Full(value=42.0, feature_size=feature_size).generate(
            batch_size=batch_size, seq_len=seq_len
        ),
        torch.full((batch_size, seq_len, feature_size), 42.0),
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_full_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    assert objects_are_equal(
        Full(value=42.0, feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len),
        torch.full((batch_size, seq_len, 3, 4), 42.0),
    )


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_full_generate_value(value: float) -> None:
    assert objects_are_equal(
        Full(value=value, feature_size=1).generate(batch_size=2, seq_len=4),
        torch.full((2, 4, 1), value),
    )


def test_full_generate_same_random_seed() -> None:
    generator = Full(value=42.0)
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_full_generate_different_random_seeds() -> None:
    # The batches are equal because the random seed is not used for this sequence creator
    generator = Full(value=42.0)
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
