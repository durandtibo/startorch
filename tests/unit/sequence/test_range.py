from __future__ import annotations

import pytest
import torch
from redcat import BatchedTensorSeq

from startorch.sequence import Arange
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


############################
#     Tests for Arange     #
############################


def test_arange_str() -> None:
    assert str(Arange()).startswith("ArangeSequenceGenerator(")


def test_arange_generate() -> None:
    assert (
        Arange()
        .generate(batch_size=2, seq_len=4)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[0], [1], [2], [3]], [[0], [1], [2], [3]]], dtype=torch.long)
            )
        )
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_arange_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = Arange().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.long


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_arange_generate_feature_size_int(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Arange(feature_size=feature_size).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.long


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_arange_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = Arange(feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.long


def test_arange_generate_same_random_seed() -> None:
    generator = Arange()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_arange_generate_different_random_seeds() -> None:
    # Should be the equal because this sequence generator is fully deterministic
    generator = Arange()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
