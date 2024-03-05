from __future__ import annotations

import pytest
import torch

from startorch.sequence import (
    Acosh,
    Asinh,
    Atanh,
    Cosh,
    Full,
    RandNormal,
    RandUniform,
    Sinh,
    Tanh,
)
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


###########################
#     Tests for Acosh     #
###########################


def test_acosh_str() -> None:
    assert str(Acosh(RandUniform())).startswith("AcoshSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_acosh_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Acosh(Full(value=1.0, feature_size=feature_size))
        .generate(batch_size=batch_size, seq_len=seq_len)
        .allclose(torch.full((batch_size, seq_len, feature_size), 0.0))
    )


def test_acosh_generate_same_random_seed() -> None:
    generator = Acosh(RandUniform(low=1.0, high=2.0))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_acosh_generate_different_random_seeds() -> None:
    generator = Acosh(RandUniform(low=1.0, high=2.0))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


###########################
#     Tests for Asinh     #
###########################


def test_asinh_str() -> None:
    assert str(Asinh(RandUniform())).startswith("AsinhSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_asinh_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Asinh(Full(value=1.0, feature_size=feature_size))
        .generate(batch_size=batch_size, seq_len=seq_len)
        .allclose(torch.full((batch_size, seq_len, feature_size), 0.881373587019543))
    )


def test_asinh_generate_same_random_seed() -> None:
    generator = Asinh(RandUniform())
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_asinh_generate_different_random_seeds() -> None:
    generator = Asinh(RandUniform())
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


###########################
#     Tests for Atanh     #
###########################


def test_atanh_str() -> None:
    assert str(Atanh(RandUniform())).startswith("AtanhSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_atanh_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Atanh(Full(value=0.42, feature_size=feature_size))
        .generate(batch_size=batch_size, seq_len=seq_len)
        .allclose(torch.full((batch_size, seq_len, feature_size), 0.44769202352742066))
    )


def test_atanh_generate_same_random_seed() -> None:
    generator = Atanh(RandUniform())
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_atanh_generate_different_random_seeds() -> None:
    generator = Atanh(RandUniform())
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


##########################
#     Tests for Cosh     #
##########################


def test_cosh_str() -> None:
    assert str(Cosh(RandUniform())).startswith("CoshSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_cosh_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Cosh(Full(value=0.0, feature_size=feature_size))
        .generate(batch_size=batch_size, seq_len=seq_len)
        .allclose(torch.full((batch_size, seq_len, feature_size), 1.0))
    )


def test_cosh_generate_same_random_seed() -> None:
    generator = Cosh(RandNormal())
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_cosh_generate_different_random_seeds() -> None:
    generator = Cosh(RandNormal())
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


##########################
#     Tests for Sinh     #
##########################


def test_sinh_str() -> None:
    assert str(Sinh(RandUniform())).startswith("SinhSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_sinh_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Sinh(Full(value=1.0, feature_size=feature_size))
        .generate(batch_size=batch_size, seq_len=seq_len)
        .allclose(torch.full((batch_size, seq_len, feature_size), 1.1752011936438014))
    )


def test_sinh_generate_same_random_seed() -> None:
    generator = Sinh(RandUniform())
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_sinh_generate_different_random_seeds() -> None:
    generator = Sinh(RandUniform())
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


##########################
#     Tests for Tanh     #
##########################


def test_tanh_str() -> None:
    assert str(Tanh(RandUniform())).startswith("TanhSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_tanh_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    assert (
        Tanh(Full(value=1.0, feature_size=feature_size))
        .generate(batch_size=batch_size, seq_len=seq_len)
        .allclose(torch.full((batch_size, seq_len, feature_size), 0.7615941559557649))
    )


def test_tanh_generate_same_random_seed() -> None:
    generator = Tanh(RandUniform())
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_tanh_generate_different_random_seeds() -> None:
    generator = Tanh(RandUniform())
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
