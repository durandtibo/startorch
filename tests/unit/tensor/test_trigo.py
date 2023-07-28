from __future__ import annotations

import torch
from pytest import mark

from startorch.tensor import Acosh, Asinh, Atanh, Cosh, Full, RandUniform, Sinh, Tanh
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


###########################
#     Tests for Acosh     #
###########################


def test_acosh_str() -> None:
    assert str(Acosh(RandUniform())).startswith("AcoshTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_acosh_generate(size: tuple[int, ...]) -> None:
    assert Acosh(Full(value=1.0)).generate(size).allclose(torch.full(size, 0.0))


def test_acosh_generate_same_random_seed() -> None:
    generator = Acosh(RandUniform(low=1.0, high=5.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_acosh_generate_different_random_seeds() -> None:
    generator = Acosh(RandUniform(low=1.0, high=5.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


###########################
#     Tests for Asinh     #
###########################


def test_asinh_str() -> None:
    assert str(Asinh(RandUniform())).startswith("AsinhTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_asinh_generate(size: tuple[int, ...]) -> None:
    assert Asinh(Full(value=1.0)).generate(size).allclose(torch.full(size, 0.881373587019543))


def test_asinh_generate_same_random_seed() -> None:
    generator = Asinh(RandUniform())
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_asinh_generate_different_random_seeds() -> None:
    generator = Asinh(RandUniform())
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


###########################
#     Tests for Atanh     #
###########################


def test_atanh_str() -> None:
    assert str(Atanh(RandUniform())).startswith("AtanhTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_atanh_generate(size: tuple[int, ...]) -> None:
    assert Atanh(Full(value=0.0)).generate(size).allclose(torch.full(size, 0.0))


def test_atanh_generate_same_random_seed() -> None:
    generator = Atanh(RandUniform())
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_atanh_generate_different_random_seeds() -> None:
    generator = Atanh(RandUniform())
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


##########################
#     Tests for Cosh     #
##########################


def test_cosh_str() -> None:
    assert str(Cosh(RandUniform())).startswith("CoshTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_cosh_generate(size: tuple[int, ...]) -> None:
    assert Cosh(Full(value=0.0)).generate(size).allclose(torch.full(size, 1.0))


def test_cosh_generate_same_random_seed() -> None:
    generator = Cosh(RandUniform(low=1.0, high=5.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_cosh_generate_different_random_seeds() -> None:
    generator = Cosh(RandUniform(low=1.0, high=5.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


##########################
#     Tests for Sinh     #
##########################


def test_sinh_str() -> None:
    assert str(Sinh(RandUniform())).startswith("SinhTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_sinh_generate(size: tuple[int, ...]) -> None:
    assert Sinh(Full(value=1.0)).generate(size).allclose(torch.full(size, 1.1752011936438014))


def test_sinh_generate_same_random_seed() -> None:
    generator = Sinh(RandUniform())
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_sinh_generate_different_random_seeds() -> None:
    generator = Sinh(RandUniform())
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


##########################
#     Tests for Tanh     #
##########################


def test_tanh_str() -> None:
    assert str(Tanh(RandUniform())).startswith("TanhTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_tanh_generate(size: tuple[int, ...]) -> None:
    assert Tanh(Full(value=1.0)).generate(size).allclose(torch.full(size, 0.7615941559557649))


def test_tanh_generate_same_random_seed() -> None:
    generator = Tanh(RandUniform())
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_tanh_generate_different_random_seeds() -> None:
    generator = Tanh(RandUniform())
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
