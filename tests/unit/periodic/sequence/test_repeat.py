from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from coola import objects_are_equal

from startorch.periodic.sequence import Repeat
from startorch.sequence import BaseSequenceGenerator, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]
DTYPES = (torch.float, torch.long)


############################
#     Tests for Repeat     #
############################


def test_repeat_str() -> None:
    assert str(Repeat(RandUniform())).startswith("RepeatPeriodicSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("period", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_repeat_generate(batch_size: int, seq_len: int, period: int, feature_size: int) -> None:
    batch = Repeat(RandUniform(feature_size=feature_size)).generate(
        batch_size=batch_size, seq_len=seq_len, period=period
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_repeat_period_3() -> None:
    batch = Repeat(RandUniform()).generate(batch_size=2, period=3, seq_len=10)
    assert isinstance(batch, torch.Tensor)
    assert objects_are_equal(batch.data[:, :3], batch.data[:, 3:6])
    assert objects_are_equal(batch.data[:, :3], batch.data[:, 6:9])


def test_repeat_period_4() -> None:
    batch = Repeat(RandUniform()).generate(batch_size=2, period=4, seq_len=10)
    assert isinstance(batch, torch.Tensor)
    assert objects_are_equal(batch.data[:, :4], batch.data[:, 4:8])
    assert objects_are_equal(batch.data[:, :2], batch.data[:, 8:])


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_dtype_dim_2(dtype: torch.dtype) -> None:
    assert (
        Repeat(
            Mock(
                spec=BaseSequenceGenerator,
                generate=Mock(return_value=torch.ones(2, 3, dtype=dtype)),
            )
        )
        .generate(batch_size=2, seq_len=6, period=3)
        .equal(torch.ones(2, 6, dtype=dtype))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_dtype_dim_3(dtype: torch.dtype) -> None:
    assert (
        Repeat(
            Mock(
                spec=BaseSequenceGenerator,
                generate=Mock(return_value=torch.ones(2, 3, 4, dtype=dtype)),
            )
        )
        .generate(batch_size=2, seq_len=6, period=3)
        .equal(torch.ones(2, 6, 4, dtype=dtype))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_repeat_dtype_dim_4(dtype: torch.dtype) -> None:
    assert (
        Repeat(
            Mock(
                spec=BaseSequenceGenerator,
                generate=Mock(return_value=torch.ones(2, 3, 4, 5, dtype=dtype)),
            )
        )
        .generate(batch_size=2, seq_len=6, period=3)
        .equal(torch.ones(2, 6, 4, 5, dtype=dtype))
    )


def test_repeat_generate_same_random_seed() -> None:
    generator = Repeat(RandUniform())
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)),
    )


def test_repeat_generate_different_random_seeds() -> None:
    generator = Repeat(RandUniform())
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, period=5, rng=get_torch_generator(2)),
    )
