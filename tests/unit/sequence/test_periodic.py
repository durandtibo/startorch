from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from batchtensor.tensor import slice_along_seq
from coola import objects_are_equal

from startorch.periodic.sequence import BasePeriodicSequenceGenerator, Repeat
from startorch.sequence import BaseSequenceGenerator, Periodic, RandUniform
from startorch.tensor import BaseTensorGenerator, RandInt
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


##############################
#     Tests for Periodic     #
##############################


def test_periodic_str() -> None:
    assert str(Periodic(sequence=RandUniform(), period=RandInt(2, 5))).startswith(
        "PeriodicSequenceGenerator("
    )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("sequence", [RandUniform(), Repeat(RandUniform())])
def test_periodic_generate(
    batch_size: int,
    seq_len: int,
    sequence: BaseSequenceGenerator | BasePeriodicSequenceGenerator,
) -> None:
    batch = Periodic(
        sequence=sequence,
        period=RandInt(2, 5),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 1)
    assert batch.dtype == torch.float


@pytest.mark.parametrize("sequence", [RandUniform(), Repeat(RandUniform())])
def test_periodic_generate_period_4(
    sequence: BaseSequenceGenerator | BasePeriodicSequenceGenerator,
) -> None:
    batch = Periodic(
        sequence=sequence,
        period=Mock(spec=BaseTensorGenerator, generate=Mock(return_value=torch.tensor([4]))),
    ).generate(batch_size=2, seq_len=10)
    assert isinstance(batch, torch.Tensor)
    assert objects_are_equal(
        slice_along_seq(batch, start=0, stop=4), slice_along_seq(batch, start=4, stop=8)
    )
    assert objects_are_equal(
        slice_along_seq(batch, start=0, stop=2), slice_along_seq(batch, start=8)
    )


def test_periodic_generate_same_random_seed() -> None:
    generator = Periodic(sequence=RandUniform(), period=RandInt(2, 5))
    assert objects_are_equal(
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1)),
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1)),
    )


def test_periodic_generate_different_random_seeds() -> None:
    generator = Periodic(sequence=RandUniform(), period=RandInt(2, 5))
    assert not objects_are_equal(
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(1)),
        generator.generate(seq_len=12, batch_size=4, rng=get_torch_generator(2)),
    )
