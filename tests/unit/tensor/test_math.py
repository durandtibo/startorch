from __future__ import annotations

import pytest
import torch
from objectory import OBJECT_TARGET

from startorch.tensor import (
    Abs,
    Add,
    AddScalar,
    Clamp,
    Div,
    Exp,
    Fmod,
    Full,
    Log,
    Mul,
    MulScalar,
    Neg,
    RandInt,
    RandNormal,
    RandUniform,
    Sqrt,
    Sub,
)
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


#########################
#     Tests for Abs     #
#########################


def test_abs_str() -> None:
    assert str(Abs(RandNormal())).startswith("AbsTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_abs_generate(size: tuple[int, ...]) -> None:
    generator = Abs(RandNormal())
    tensor = generator.generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0


def test_abs_generate_fixed_value() -> None:
    assert Abs(Full(-1.0)).generate(size=(2, 4)).equal(torch.ones(2, 4))


def test_abs_generate_same_random_seed() -> None:
    generator = Abs(RandNormal())
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_abs_generate_different_random_seeds() -> None:
    generator = Abs(RandNormal())
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


###############################
#     Tests for AddScalar     #
###############################


def test_add_scalar_str() -> None:
    assert str(AddScalar(RandUniform(), value=1.0)).startswith("AddScalarTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_add_scalar_generate(size: tuple[int, ...]) -> None:
    generator = AddScalar(RandUniform(), value=1.0)
    tensor = generator.generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 1.0
    assert tensor.max() < 2.0


def test_add_scalar_generate_2() -> None:
    assert (
        AddScalar(Full(1.0), 2.0)
        .generate(size=(2, 4, 1))
        .equal(torch.tensor([[[3.0], [3.0], [3.0], [3.0]], [[3.0], [3.0], [3.0], [3.0]]]))
    )


def test_add_scalar_generate_3() -> None:
    assert (
        AddScalar(Full(1.0), -3.0)
        .generate(size=(2, 4, 1))
        .equal(torch.tensor([[[-2.0], [-2.0], [-2.0], [-2.0]], [[-2.0], [-2.0], [-2.0], [-2.0]]]))
    )


def test_add_scalar_generate_same_random_seed() -> None:
    generator = AddScalar(RandUniform(), value=1.0)
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_add_scalar_generate_different_random_seeds() -> None:
    generator = AddScalar(RandUniform(), value=1.0)
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#########################
#     Tests for Add     #
#########################


def test_add_str() -> None:
    assert str(Add((RandUniform(), RandUniform()))).startswith("AddTensorGenerator(")


def test_add_2_tensors() -> None:
    generator = Add((RandUniform(), {OBJECT_TARGET: "startorch.tensor.RandUniform"}))
    assert len(generator._generators) == 2
    assert isinstance(generator._generators[0], RandUniform)
    assert isinstance(generator._generators[1], RandUniform)


def test_add_3_tensors() -> None:
    generator = Add(
        (
            RandUniform(),
            RandNormal(),
            {OBJECT_TARGET: "startorch.tensor.RandUniform"},
        )
    )
    assert len(generator._generators) == 3
    assert isinstance(generator._generators[0], RandUniform)
    assert isinstance(generator._generators[1], RandNormal)
    assert isinstance(generator._generators[2], RandUniform)


def test_add_tensors_empty() -> None:
    with pytest.raises(ValueError, match="No tensor generator."):
        Add(generators=[])


@pytest.mark.parametrize("size", SIZES)
def test_add_generate(size: tuple[int, ...]) -> None:
    tensor = Add((RandUniform(), RandUniform())).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0
    assert tensor.max() < 2.0


def test_add_generate_fixed_values() -> None:
    assert (
        Add(
            (Full(1.0), Full(2.0), Full(5.0)),
        )
        .generate(size=(4, 12))
        .equal(torch.full((4, 12), fill_value=8.0, dtype=torch.float))
    )


def test_add_generate_same_random_seed() -> None:
    generator = Add((RandUniform(), RandUniform()))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_add_generate_different_random_seeds() -> None:
    generator = Add((RandUniform(), RandUniform()))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


###########################
#     Tests for Clamp     #
###########################


def test_clamp_str() -> None:
    assert str(Clamp(RandNormal(), min_value=-2, max_value=2)).startswith("ClampTensorGenerator(")


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
def test_clamp_min_value(min_value: float) -> None:
    assert Clamp(RandNormal(), min_value=min_value, max_value=None)._min_value == min_value


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_clamp_max_value(max_value: float) -> None:
    assert Clamp(RandNormal(), min_value=None, max_value=max_value)._max_value == max_value


def test_clamp_incorrect_min_max() -> None:
    with pytest.raises(ValueError, match="`min_value` and `max_value` cannot be both None"):
        Clamp(RandNormal(), min_value=None, max_value=None)


@pytest.mark.parametrize("size", SIZES)
def test_clamp_generate(size: tuple[int, ...]) -> None:
    tensor = Clamp(
        RandNormal(),
        min_value=-2,
        max_value=2,
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= -2.0
    assert tensor.max() <= 2.0


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_clamp_generate_min_max_float(min_value: float, max_value: float) -> None:
    tensor = Clamp(
        RandNormal(),
        min_value=min_value,
        max_value=max_value,
    ).generate(size=(4, 12))
    assert tensor.min() >= min_value
    assert tensor.max() <= max_value


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_clamp_generate_min_max_long(min_value: int, max_value: int) -> None:
    tensor = Clamp(RandInt(low=-5, high=20), min_value=min_value, max_value=max_value).generate(
        size=(4, 12)
    )
    assert tensor.min() >= min_value
    assert tensor.max() <= max_value


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
def test_clamp_generate_only_min_value(min_value: float) -> None:
    assert (
        Clamp(RandNormal(), min_value=min_value, max_value=None).generate(size=(4, 12)).min()
        >= min_value
    )


@pytest.mark.parametrize("max_value", [-1.0, -2.0])
def test_clamp_generate_only_max_value(max_value: float) -> None:
    assert (
        Clamp(RandNormal(), min_value=None, max_value=max_value).generate(size=(4, 12)).max()
        <= max_value
    )


def test_clamp_generate_same_random_seed() -> None:
    generator = Clamp(RandNormal(), min_value=-2, max_value=2)
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_clamp_generate_different_random_seeds() -> None:
    generator = Clamp(RandNormal(), min_value=-2, max_value=2)
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#########################
#     Tests for Div     #
#########################


def test_div_str() -> None:
    assert str(Div(RandUniform(low=0.1, high=2.0), RandUniform(low=0.1, high=2.0))).startswith(
        "DivTensorGenerator("
    )


@pytest.mark.parametrize("size", SIZES)
def test_div_generate(size: tuple[int, ...]) -> None:
    tensor = Div(
        dividend=RandUniform(low=0.1, high=2.0),
        divisor=RandUniform(low=0.1, high=2.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


@pytest.mark.parametrize("size", SIZES)
def test_div_generate_rounding_mode_default(size: tuple[int, ...]) -> None:
    assert Div(dividend=Full(3.0), divisor=Full(2.0)).generate(size).equal(torch.full(size, 1.5))


@pytest.mark.parametrize("size", SIZES)
def test_div_generate_rounding_mode_floor(size: tuple[int, ...]) -> None:
    assert (
        Div(dividend=Full(3.0), divisor=Full(2.0), rounding_mode="floor")
        .generate(size)
        .equal(torch.ones(size))
    )


def test_div_generate_same_random_seed() -> None:
    generator = Div(dividend=RandUniform(low=0.1, high=2.0), divisor=RandUniform(low=0.1, high=2.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_div_generate_different_random_seeds() -> None:
    generator = Div(dividend=RandUniform(low=0.1, high=2.0), divisor=RandUniform(low=0.1, high=2.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#########################
#     Tests for Exp     #
#########################


def test_exp_str() -> None:
    assert str(Exp(RandUniform())).startswith("ExpTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_exp_generate(size: tuple[int, ...]) -> None:
    assert Exp(Full(value=0.0)).generate(size).equal(torch.ones(size, dtype=torch.float))


def test_exp_generate_same_random_seed() -> None:
    generator = Exp(RandUniform())
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_exp_generate_different_random_seeds() -> None:
    generator = Exp(RandUniform())
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


##########################
#     Tests for Fmod     #
##########################


def test_fmod_str() -> None:
    assert str(
        Fmod(
            dividend=RandUniform(low=-100.0, high=100.0),
            divisor=RandUniform(low=1.0, high=10.0),
        )
    ).startswith("FmodTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_fmod_generate_divisor_generator(size: tuple[int, ...]) -> None:
    assert (
        Fmod(dividend=Full(5.0), divisor=Full(10.0))
        .generate(size)
        .equal(torch.full(size, 5.0, dtype=torch.float))
    )


@pytest.mark.parametrize("size", SIZES)
def test_fmod_generate_divisor_number(size: tuple[int, ...]) -> None:
    assert (
        Fmod(dividend=Full(5.0), divisor=10.0)
        .generate(size)
        .equal(torch.full(size, 5.0, dtype=torch.float))
    )


def test_fmod_generate_same_random_seed() -> None:
    generator = Fmod(
        dividend=RandUniform(low=-100.0, high=100.0), divisor=RandUniform(low=1.0, high=10.0)
    )
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_fmod_generate_different_random_seeds() -> None:
    generator = Fmod(
        dividend=RandUniform(low=-100.0, high=100.0), divisor=RandUniform(low=1.0, high=10.0)
    )
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#########################
#     Tests for Log     #
#########################


def test_log_str() -> None:
    assert str(Log(RandUniform(low=0.1, high=2.0))).startswith("LogTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_log_generate(size: tuple[int, ...]) -> None:
    assert Log(Full(value=1.0)).generate(size).equal(torch.zeros(size, dtype=torch.float))


def test_log_generate_same_random_seed() -> None:
    generator = Log(RandUniform(low=0.1, high=2.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_log_generate_different_random_seeds() -> None:
    generator = Log(RandUniform(low=0.1, high=2.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#########################
#     Tests for Mul     #
#########################


def test_mul_str() -> None:
    assert str(Mul((RandUniform(), RandUniform()))).startswith("MulTensorGenerator(")


def test_mul_2_tensors() -> None:
    generator = Mul((RandUniform(), {OBJECT_TARGET: "startorch.tensor.RandUniform"}))
    assert len(generator._generators) == 2
    assert isinstance(generator._generators[0], RandUniform)
    assert isinstance(generator._generators[1], RandUniform)


def test_mul_3_tensors() -> None:
    generator = Mul(
        (
            RandUniform(),
            RandNormal(),
            {OBJECT_TARGET: "startorch.tensor.RandUniform"},
        )
    )
    assert len(generator._generators) == 3
    assert isinstance(generator._generators[0], RandUniform)
    assert isinstance(generator._generators[1], RandNormal)
    assert isinstance(generator._generators[2], RandUniform)


def test_mul_tensors_empty() -> None:
    with pytest.raises(ValueError, match="No tensor generator."):
        Mul(generators=[])


@pytest.mark.parametrize("size", SIZES)
def test_mul_generate(size: tuple[int, ...]) -> None:
    tensor = Mul((RandUniform(), RandUniform())).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_mul_generate_weight() -> None:
    assert (
        Mul((Full(1.0), Full(2.0), Full(5.0)))
        .generate(size=(2, 5))
        .equal(torch.full((2, 5), fill_value=10.0, dtype=torch.float))
    )


def test_mul_generate_same_random_seed() -> None:
    generator = Mul((RandUniform(), RandUniform()))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_mul_generate_different_random_seeds() -> None:
    generator = Mul((RandUniform(), RandUniform()))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


###############################
#     Tests for MulScalar     #
###############################


def test_mul_scalar_str() -> None:
    assert str(MulScalar(RandUniform(), value=1.0)).startswith("MulScalarTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_mul_scalar_generate(size: tuple[int, ...]) -> None:
    generator = MulScalar(RandUniform(), value=2.0)
    tensor = generator.generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0
    assert tensor.max() < 2.0


def test_mul_scalar_generate_2() -> None:
    assert (
        MulScalar(Full(1.0), 2.0)
        .generate(size=(2, 4, 1))
        .equal(torch.tensor([[[2.0], [2.0], [2.0], [2.0]], [[2.0], [2.0], [2.0], [2.0]]]))
    )


def test_mul_scalar_generate_3() -> None:
    assert (
        MulScalar(Full(1.0), -3.0)
        .generate(size=(2, 4, 1))
        .equal(torch.tensor([[[-3.0], [-3.0], [-3.0], [-3.0]], [[-3.0], [-3.0], [-3.0], [-3.0]]]))
    )


def test_mul_scalar_generate_same_random_seed() -> None:
    generator = MulScalar(RandUniform(), value=1.0)
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_mul_scalar_generate_different_random_seeds() -> None:
    generator = MulScalar(RandUniform(), value=1.0)
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#########################
#     Tests for Neg     #
#########################


def test_neg_str() -> None:
    assert str(Neg(RandUniform())).startswith("NegTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_neg_generate(size: tuple[int, ...]) -> None:
    generator = Neg(RandUniform())
    tensor = generator.generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_neg_generate_fixed_value() -> None:
    assert (
        Neg(Full(1.0))
        .generate(size=(2, 4, 1))
        .equal(torch.tensor([[[-1.0], [-1.0], [-1.0], [-1.0]], [[-1.0], [-1.0], [-1.0], [-1.0]]]))
    )


def test_neg_generate_same_random_seed() -> None:
    generator = Neg(RandUniform())
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_neg_generate_different_random_seeds() -> None:
    generator = Neg(RandUniform())
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


##########################
#     Tests for Sqrt     #
##########################


def test_sqrt_str() -> None:
    assert str(Sqrt(RandUniform(low=0.1, high=2.0))).startswith("SqrtTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_sqrt_generate(size: tuple[int, ...]) -> None:
    assert Sqrt(Full(value=4.0)).generate(size).equal(torch.full(size, 2.0, dtype=torch.float))


def test_sqrt_generate_same_random_seed() -> None:
    generator = Sqrt(RandUniform(low=0.1, high=2.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_sqrt_generate_different_random_seeds() -> None:
    generator = Sqrt(RandUniform(low=0.1, high=2.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#########################
#     Tests for Sub     #
#########################


def test_sub_str() -> None:
    assert str(Sub(tensor1=RandUniform(), tensor2=RandUniform())).startswith("SubTensorGenerator(")


@pytest.mark.parametrize("size", SIZES)
def test_sub_generate(size: tuple[int, ...]) -> None:
    tensor = Sub(tensor1=RandUniform(), tensor2=RandUniform()).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_sub_generate_same_random_seed() -> None:
    generator = Sub(tensor1=RandUniform(), tensor2=RandUniform())
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_sub_generate_different_random_seeds() -> None:
    generator = Sub(tensor1=RandUniform(), tensor2=RandUniform())
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
