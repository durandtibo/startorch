from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.sequence import TensorSequence
from startorch.tensor import Full, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


####################################
#     Tests for TensorSequence     #
####################################


def test_tensor_sequence_str() -> None:
    assert str(TensorSequence(RandUniform())).startswith("TensorSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_tensor_sequence_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    assert objects_are_equal(
        TensorSequence(Full(1.0)).generate(batch_size=batch_size, seq_len=seq_len),
        torch.ones(batch_size, seq_len),
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_tensor_sequence_generate_int(batch_size: int, seq_len: int) -> None:
    assert objects_are_equal(
        TensorSequence(Full(1.0), feature_size=4).generate(batch_size=batch_size, seq_len=seq_len),
        torch.ones(batch_size, seq_len, 4),
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_tensor_sequence_generate_tuple(batch_size: int, seq_len: int) -> None:
    assert objects_are_equal(
        TensorSequence(Full(1.0), feature_size=(3, 4)).generate(
            batch_size=batch_size, seq_len=seq_len
        ),
        torch.ones(batch_size, seq_len, 3, 4),
    )


def test_tensor_sequence_generate_same_random_seed() -> None:
    generator = TensorSequence(RandUniform())
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_tensor_sequence_generate_different_random_seeds() -> None:
    generator = TensorSequence(RandUniform())
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
