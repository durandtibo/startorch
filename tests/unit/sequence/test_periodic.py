from __future__ import annotations

from unittest.mock import Mock

import torch
from pytest import mark
from redcat import BatchedTensorSeq

from startorch.periodic.sequence import BasePeriodicSequenceGenerator, Repeat
from startorch.sequence import BaseSequenceGenerator, Periodic, RandUniform
from startorch.tensor import BaseTensorGenerator, RandInt
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2)


##############################
#     Tests for Periodic     #
##############################


def test_periodic_str() -> None:
    assert str(Periodic(sequence=RandUniform(), period=RandInt(2, 5))).startswith(
        "PeriodicSequenceGenerator("
    )


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("sequence", (RandUniform(), Repeat(RandUniform())))
def test_periodic_generate(
    batch_size: int,
    seq_len: int,
    sequence: BaseSequenceGenerator | BasePeriodicSequenceGenerator,
) -> None:
    batch = Periodic(
        sequence=sequence,
        period=RandInt(2, 5),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float


@mark.parametrize("generator", (RandUniform(), Repeat(RandUniform())))
def test_periodic_generate_period_4(
    sequence: BaseSequenceGenerator | BasePeriodicSequenceGenerator,
) -> None:
    batch = Periodic(
        sequence=sequence,
        period=Mock(spec=BaseTensorGenerator, generate=Mock(return_value=torch.tensor([4]))),
    ).generate(batch_size=2, seq_len=10)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == 2
    assert batch.seq_len == 10
    assert batch.slice_along_seq(0, 4).equal(batch.slice_along_seq(4, 8))
    assert batch.slice_along_seq(0, 2).equal(batch.slice_along_seq(8))


def test_periodic_generate_same_random_seed() -> None:
    generator = Periodic(sequence=RandUniform(), period=RandInt(2, 5))
    assert generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1)).equal(
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1))
    )


def test_periodic_generate_different_random_seeds() -> None:
    generator = Periodic(sequence=RandUniform(), period=RandInt(2, 5))
    assert not generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1)).equal(
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(2))
    )
