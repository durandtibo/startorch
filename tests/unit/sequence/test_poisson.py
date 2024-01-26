from __future__ import annotations

import pytest
import torch
from redcat import BatchedTensorSeq

from startorch.sequence import Constant, Poisson, RandPoisson, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


#############################
#     Tests for Poisson     #
#############################


def test_poisson_str() -> None:
    assert str(Poisson(RandUniform())).startswith("PoissonSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_poisson_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = Poisson(Constant(RandUniform(feature_size=feature_size))).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


def test_poisson_generate_same_random_seed() -> None:
    generator = Poisson(Constant(RandUniform()))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_poisson_generate_different_random_seeds() -> None:
    generator = Poisson(Constant(RandUniform()))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


@pytest.mark.parametrize("generator", [Poisson.generate_uniform_rate()])
def test_poisson_generate_predefined_generators(generator: Poisson) -> None:
    batch = generator.generate(batch_size=4, seq_len=12)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == 4
    assert batch.seq_len == 12
    assert batch.data.shape == (4, 12, 1)
    assert batch.data.dtype == torch.float


#################################
#     Tests for RandPoisson     #
#################################


def test_rand_poisson_str() -> None:
    assert str(RandPoisson()).startswith("RandPoissonSequenceGenerator(")


@pytest.mark.parametrize("rate", [1.0, 2.0])
def test_rand_poisson_rate(rate: float) -> None:
    assert RandPoisson(rate=rate)._rate == rate


def test_rand_poisson_rate_default() -> None:
    assert RandPoisson()._rate == 1.0


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_rand_poisson_rate_incorrect(rate: float) -> None:
    with pytest.raises(ValueError, match="rate has to be greater than 0"):
        RandPoisson(rate=rate)


def test_rand_poisson_feature_size_default() -> None:
    assert RandPoisson()._feature_size == (1,)


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_poisson_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandPoisson().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_rand_poisson_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandPoisson(feature_size=feature_size).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_rand_poisson_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandPoisson(feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.float
    assert batch.min() >= 0.0


def test_rand_poisson_generate_same_random_seed() -> None:
    generator = RandPoisson()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_poisson_generate_different_random_seeds() -> None:
    generator = RandPoisson()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
