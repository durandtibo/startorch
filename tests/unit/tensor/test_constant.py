from __future__ import annotations

import torch
from pytest import mark

from startorch.tensor import Full
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


##########################
#     Tests for Full     #
##########################


def test_full_str() -> None:
    assert str(Full(value=42)).startswith("FullTensorGenerator(")


@mark.parametrize("value", (1.2, 42))
def test_full_value(value: int) -> None:
    assert Full(value=value)._value == value


@mark.parametrize("dtype", (torch.float, torch.long))
def test_full_dtype(dtype: torch.dtype) -> None:
    assert Full(value=42, dtype=dtype)._dtype == dtype


def test_full_dtype_default() -> None:
    assert Full(value=42)._dtype is None


@mark.parametrize("size", SIZES)
def test_full_generate_dtype_default(size: tuple[int, ...]) -> None:
    assert Full(1).generate(size).equal(torch.full(size, 1, dtype=torch.float))


@mark.parametrize("size", SIZES)
def test_full_generate_dtype_float(size: tuple[int, ...]) -> None:
    assert Full(1, dtype=torch.float).generate(size).equal(torch.full(size, 1, dtype=torch.float))


@mark.parametrize("size", SIZES)
def test_full_generate_dtype_long(size: tuple[int, ...]) -> None:
    assert Full(1, dtype=torch.long).generate(size).equal(torch.ones(size, dtype=torch.long))


def test_full_generate_same_random_seed() -> None:
    generator = Full(value=42.0)
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_full_generate_different_random_seeds() -> None:
    # The batches are equal because the random seed is not used for this sequence creator
    generator = Full(value=42.0)
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
