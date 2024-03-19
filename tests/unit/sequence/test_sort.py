from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from coola import objects_are_equal

from startorch.sequence import RandUniform, Sort
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


##########################
#     Tests for Sort     #
##########################


def test_sort_str() -> None:
    assert str(Sort(RandUniform())).startswith("SortSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_sort_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Sort(RandUniform(feature_size=feature_size)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float


@patch(
    "startorch.sequence.RandUniform.generate",
    lambda *args, **kwargs: torch.tensor(  # noqa: ARG005
        [[[2.0], [4.0], [1.0], [3.0]], [[3.0], [2.0], [1.0], [2.0]]], dtype=torch.float
    ),
)
def test_sort_generate_descending_false() -> None:
    assert objects_are_equal(
        Sort(RandUniform(feature_size=1)).generate(batch_size=2, seq_len=4),
        torch.tensor(
            [[[1.0], [2.0], [3.0], [4.0]], [[1.0], [2.0], [2.0], [3.0]]], dtype=torch.float
        ),
    )


@patch(
    "startorch.sequence.RandUniform.generate",
    lambda *args, **kwargs: torch.tensor(  # noqa: ARG005
        [[[2.0], [4.0], [1.0], [3.0]], [[3.0], [2.0], [1.0], [2.0]]], dtype=torch.float
    ),
)
def test_sort_generate_descending_true() -> None:
    assert objects_are_equal(
        Sort(RandUniform(feature_size=1), descending=True).generate(batch_size=2, seq_len=4),
        torch.tensor(
            [[[4.0], [3.0], [2.0], [1.0]], [[3.0], [2.0], [2.0], [1.0]]], dtype=torch.float
        ),
    )


def test_sort_generate_same_random_seed() -> None:
    generator = Sort(RandUniform())
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_sort_generate_different_random_seeds() -> None:
    generator = Sort(RandUniform())
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
