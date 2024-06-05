from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.sequence import VanillaSequenceGenerator
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


##############################################
#     Tests for VanillaSequenceGenerator     #
##############################################


def test_vanilla_str() -> None:
    assert str(VanillaSequenceGenerator(torch.arange(40).view(4, 10))).startswith(
        "VanillaSequenceGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
def test_vanilla_batch_size(batch_size: int) -> None:
    assert VanillaSequenceGenerator(torch.ones(batch_size, 3))._batch_size == batch_size


@pytest.mark.parametrize("seq_len", SIZES)
def test_vanilla_seq_len(seq_len: int) -> None:
    assert VanillaSequenceGenerator(torch.ones(4, seq_len))._seq_len == seq_len


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_vanilla_lower_batch_size(batch_size: int) -> None:
    out = VanillaSequenceGenerator(torch.ones(4, 10)).generate(batch_size=batch_size, seq_len=10)
    assert objects_are_equal(out, torch.ones(batch_size, 10))


def test_vanilla_larger_batch_size() -> None:
    generator = VanillaSequenceGenerator(torch.ones(4, 10))
    with pytest.raises(RuntimeError, match="Incorrect batch_size: 11."):
        generator.generate(batch_size=11, seq_len=10)


@pytest.mark.parametrize("seq_len", [1, 2, 4, 10])
def test_vanilla_lower_seq_len(seq_len: int) -> None:
    out = VanillaSequenceGenerator(torch.ones(4, 10)).generate(batch_size=4, seq_len=seq_len)
    assert objects_are_equal(out, torch.ones(4, seq_len))


def test_vanilla_larger_seq_len() -> None:
    generator = VanillaSequenceGenerator(torch.ones(4, 10))
    with pytest.raises(RuntimeError, match="Incorrect seq_len: 11."):
        generator.generate(batch_size=4, seq_len=11)


def test_vanilla_generate_same_random_seed() -> None:
    generator = VanillaSequenceGenerator(torch.randn(10, 20))
    assert objects_are_equal(
        generator.generate(batch_size=6, seq_len=16, rng=get_torch_generator(1)),
        generator.generate(batch_size=6, seq_len=16, rng=get_torch_generator(1)),
    )


def test_vanilla_generate_different_random_seeds() -> None:
    generator = VanillaSequenceGenerator(torch.randn(10, 20))
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        generator.generate(batch_size=6, seq_len=16, rng=get_torch_generator(1)),
        generator.generate(batch_size=6, seq_len=16, rng=get_torch_generator(2)),
    )
