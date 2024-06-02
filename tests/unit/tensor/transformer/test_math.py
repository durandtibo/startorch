from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from startorch.tensor.transformer import (
    AbsTensorTransformer,
    CeilTensorTransformer,
    ClampTensorTransformer,
    Expm1TensorTransformer,
    ExpTensorTransformer,
    FloorTensorTransformer,
    FracTensorTransformer,
    Log1pTensorTransformer,
    LogitTensorTransformer,
    LogTensorTransformer,
    PowTensorTransformer,
    RoundTensorTransformer,
    RsqrtTensorTransformer,
    SqrtTensorTransformer,
)
from startorch.utils.seed import get_torch_generator

##########################################
#     Tests for AbsTensorTransformer     #
##########################################


def test_abs_str() -> None:
    assert str(AbsTensorTransformer()).startswith("AbsTensorTransformer(")


def test_abs_transform() -> None:
    assert objects_are_allclose(
        AbsTensorTransformer().transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])),
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )


def test_abs_transform_same_random_seed() -> None:
    transformer = AbsTensorTransformer()
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_abs_transform_different_random_seeds() -> None:
    transformer = AbsTensorTransformer()
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


###########################################
#     Tests for CeilTensorTransformer     #
###########################################


def test_ceil_str() -> None:
    assert str(CeilTensorTransformer()).startswith("CeilTensorTransformer(")


def test_ceil_transform() -> None:
    assert objects_are_allclose(
        CeilTensorTransformer().transform(torch.tensor([[-0.6, -1.4, 2.2], [-1.1, 0.5, 0.2]])),
        torch.tensor([[-0.0, -1.0, 3.0], [-1.0, 1.0, 1.0]]),
    )


def test_ceil_transform_same_random_seed() -> None:
    transformer = CeilTensorTransformer()
    tensor = torch.tensor([[-0.6, -1.4, 2.2], [-1.1, 0.5, 0.2]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_ceil_transform_different_random_seeds() -> None:
    transformer = CeilTensorTransformer()
    tensor = torch.tensor([[-0.6, -1.4, 2.2], [-1.1, 0.5, 0.2]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


############################################
#     Tests for ClampTensorTransformer     #
############################################


def test_clamp_str() -> None:
    assert str(ClampTensorTransformer(min=-2, max=2)).startswith("ClampTensorTransformer(")


@pytest.mark.parametrize("min_value", [-1.0, -2.0])
def test_clamp_min(min_value: float) -> None:
    assert ClampTensorTransformer(min=min_value, max=None)._min == min_value


@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_clamp_max(max_value: float) -> None:
    assert ClampTensorTransformer(min=None, max=max_value)._max == max_value


def test_clamp_incorrect_min_max() -> None:
    with pytest.raises(ValueError, match="`min` and `max` cannot be both None"):
        ClampTensorTransformer(min=None, max=None)


def test_clamp_transform() -> None:
    out = ClampTensorTransformer(
        min=-2,
        max=2,
    ).transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]))
    assert objects_are_equal(out, torch.tensor([[1.0, -2.0, 2.0], [-2.0, 2.0, -2.0]]))


def test_clamp_transform_only_min_value() -> None:
    out = ClampTensorTransformer(min=-1, max=None).transform(
        torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    )
    assert objects_are_equal(out, torch.tensor([[1.0, -1.0, 3.0], [-1.0, 5.0, -1.0]]))


def test_clamp_transform_only_max_value() -> None:
    out = ClampTensorTransformer(min=None, max=-1).transform(
        torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    )
    assert objects_are_equal(out, torch.tensor([[-1.0, -2.0, -1.0], [-4.0, -1.0, -6.0]]))


def test_clamp_transform_same_random_seed() -> None:
    transformer = ClampTensorTransformer(min=-2, max=2)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_clamp_transform_different_random_seeds() -> None:
    transformer = ClampTensorTransformer(min=-2, max=2)
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


##########################################
#     Tests for ExpTensorTransformer     #
##########################################


def test_exp_str() -> None:
    assert str(ExpTensorTransformer()).startswith("ExpTensorTransformer(")


def test_exp_transform() -> None:
    assert objects_are_allclose(
        ExpTensorTransformer().transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])),
        torch.tensor(
            [
                [2.7182817459106445, 0.1353352814912796, 20.08553695678711],
                [0.018315639346837997, 148.4131622314453, 0.0024787522852420807],
            ]
        ),
    )


def test_exp_transform_same_random_seed() -> None:
    transformer = ExpTensorTransformer()
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_exp_transform_different_random_seeds() -> None:
    transformer = ExpTensorTransformer()
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


############################################
#     Tests for Expm1TensorTransformer     #
############################################


def test_expm1_str() -> None:
    assert str(Expm1TensorTransformer()).startswith("Expm1TensorTransformer(")


def test_expm1_transform() -> None:
    assert objects_are_allclose(
        Expm1TensorTransformer().transform(torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])),
        torch.tensor(
            [
                [1.718281865119934, -0.8646647334098816, 19.08553695678711],
                [-0.9816843867301941, 147.4131622314453, -0.9975212216377258],
            ]
        ),
    )


def test_expm1_transform_same_random_seed() -> None:
    transformer = Expm1TensorTransformer()
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_expm1_transform_different_random_seeds() -> None:
    transformer = Expm1TensorTransformer()
    tensor = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


############################################
#     Tests for FloorTensorTransformer     #
############################################


def test_floor_str() -> None:
    assert str(FloorTensorTransformer()).startswith("FloorTensorTransformer(")


def test_floor_transform() -> None:
    assert objects_are_allclose(
        FloorTensorTransformer().transform(torch.tensor([[-0.6, -1.4, 2.2], [-1.1, 0.5, 0.2]])),
        torch.tensor([[-1.0, -2.0, 2.0], [-2.0, 0.0, 0.0]]),
    )


def test_floor_transform_same_random_seed() -> None:
    transformer = FloorTensorTransformer()
    tensor = torch.tensor([[-0.6, -1.4, 2.2], [-1.1, 0.5, 0.2]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_floor_transform_different_random_seeds() -> None:
    transformer = FloorTensorTransformer()
    tensor = torch.tensor([[-0.6, -1.4, 2.2], [-1.1, 0.5, 0.2]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


###########################################
#     Tests for FracTensorTransformer     #
###########################################


def test_frac_str() -> None:
    assert str(FracTensorTransformer()).startswith("FracTensorTransformer(")


def test_frac_transform() -> None:
    assert objects_are_allclose(
        FracTensorTransformer().transform(torch.tensor([[-0.6, -1.4, 2.2], [-1.1, 0.5, 0.2]])),
        torch.tensor([[-0.6, -0.4, 0.2], [-0.1, 0.5, 0.2]]),
    )


def test_frac_transform_same_random_seed() -> None:
    transformer = FracTensorTransformer()
    tensor = torch.tensor([[-0.6, -1.4, 2.2], [-1.1, 0.5, 0.2]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_frac_transform_different_random_seeds() -> None:
    transformer = FracTensorTransformer()
    tensor = torch.tensor([[-0.6, -1.4, 2.2], [-1.1, 0.5, 0.2]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


##########################################
#     Tests for LogTensorTransformer     #
##########################################


def test_log_str() -> None:
    assert str(LogTensorTransformer()).startswith("LogTensorTransformer(")


def test_log_transform() -> None:
    assert objects_are_allclose(
        LogTensorTransformer().transform(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        torch.tensor(
            [
                [0.0, 0.6931471824645996, 1.0986123085021973],
                [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
            ]
        ),
    )


def test_log_transform_same_random_seed() -> None:
    transformer = LogTensorTransformer()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_log_transform_different_random_seeds() -> None:
    transformer = LogTensorTransformer()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


############################################
#     Tests for Log1pTensorTransformer     #
############################################


def test_log1p_str() -> None:
    assert str(Log1pTensorTransformer()).startswith("Log1pTensorTransformer(")


def test_log1p_transform() -> None:
    assert objects_are_allclose(
        Log1pTensorTransformer().transform(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])),
        torch.tensor(
            [
                [0.0, 0.6931471824645996, 1.0986123085021973],
                [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
            ]
        ),
    )


def test_log1p_transform_same_random_seed() -> None:
    transformer = Log1pTensorTransformer()
    tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_log1p_transform_different_random_seeds() -> None:
    transformer = Log1pTensorTransformer()
    tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


############################################
#     Tests for LogitTensorTransformer     #
############################################


def test_logit_str() -> None:
    assert str(LogitTensorTransformer()).startswith("LogitTensorTransformer(")


def test_logit_transform() -> None:
    assert objects_are_allclose(
        LogitTensorTransformer().transform(torch.tensor([[0.6, 0.4, 0.3], [0.1, 0.5, 0.2]])),
        torch.tensor(
            [
                [0.40546518564224243, -0.40546515583992004, -0.8472977876663208],
                [-2.1972246170043945, 0.0, -1.3862943649291992],
            ]
        ),
    )


def test_logit_transform_same_random_seed() -> None:
    transformer = LogitTensorTransformer()
    tensor = torch.tensor([[0.6, 0.4, 0.3], [0.1, 0.5, 0.2]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_logit_transform_different_random_seeds() -> None:
    transformer = LogitTensorTransformer()
    tensor = torch.tensor([[0.6, 0.4, 0.3], [0.1, 0.5, 0.2]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


##########################################
#     Tests for PowTensorTransformer     #
##########################################


def test_pow_str() -> None:
    assert str(PowTensorTransformer(exponent=2)).startswith("PowTensorTransformer(")


def test_pow_transform_exponent_2() -> None:
    assert objects_are_allclose(
        PowTensorTransformer(exponent=2).transform(
            torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        ),
        torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]),
    )


def test_pow_transform_exponent_3() -> None:
    assert objects_are_allclose(
        PowTensorTransformer(exponent=3).transform(
            torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        ),
        torch.tensor([[0.0, 1.0, 8.0], [27.0, 64.0, 125.0]]),
    )


def test_pow_transform_same_random_seed() -> None:
    transformer = PowTensorTransformer(exponent=2)
    tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_pow_transform_different_random_seeds() -> None:
    transformer = PowTensorTransformer(exponent=2)
    tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


############################################
#     Tests for RoundTensorTransformer     #
############################################


def test_round_str() -> None:
    assert str(RoundTensorTransformer()).startswith("RoundTensorTransformer(")


def test_round_transform() -> None:
    assert objects_are_allclose(
        RoundTensorTransformer().transform(torch.tensor([[-0.6, -1.4, 2.2], [-1.1, 0.7, 0.2]])),
        torch.tensor([[-1.0, -1.0, 2.0], [-1.0, 1.0, 0.0]]),
    )


def test_round_transform_decimals_2() -> None:
    assert objects_are_allclose(
        RoundTensorTransformer(decimals=2).transform(
            torch.tensor([[-0.6666, -1.4444, 2.2222], [-1.1111, 0.7777, 0.2222]])
        ),
        torch.tensor([[-0.67, -1.44, 2.22], [-1.11, 0.78, 0.22]]),
    )


def test_round_transform_same_random_seed() -> None:
    transformer = RoundTensorTransformer()
    tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_round_transform_different_random_seeds() -> None:
    transformer = RoundTensorTransformer()
    tensor = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


############################################
#     Tests for RsqrtTensorTransformer     #
############################################


def test_rsqrt_str() -> None:
    assert str(RsqrtTensorTransformer()).startswith("RsqrtTensorTransformer(")


def test_rsqrt_transform() -> None:
    assert objects_are_allclose(
        RsqrtTensorTransformer().transform(torch.tensor([[1.0, 4.0, 16.0], [1.0, 2.0, 3.0]])),
        torch.tensor([[1.0, 0.5, 0.25], [1.0, 0.7071067690849304, 0.5773502588272095]]),
    )


def test_rsqrt_transform_same_random_seed() -> None:
    transformer = RsqrtTensorTransformer()
    tensor = torch.tensor([[1.0, 4.0, 16.0], [1.0, 2.0, 3.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_rsqrt_transform_different_random_seeds() -> None:
    transformer = RsqrtTensorTransformer()
    tensor = torch.tensor([[1.0, 4.0, 16.0], [1.0, 2.0, 3.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )


###########################################
#     Tests for SqrtTensorTransformer     #
###########################################


def test_sqrt_str() -> None:
    assert str(SqrtTensorTransformer()).startswith("SqrtTensorTransformer(")


def test_sqrt_transform() -> None:
    assert objects_are_allclose(
        SqrtTensorTransformer().transform(torch.tensor([[0.0, 4.0, 16.0], [1.0, 2.0, 3.0]])),
        torch.tensor([[0.0, 2.0, 4.0], [1.0, 1.4142135623730951, 1.7320508075688772]]),
    )


def test_sqrt_transform_same_random_seed() -> None:
    transformer = SqrtTensorTransformer()
    tensor = torch.tensor([[0.0, 4.0, 16.0], [1.0, 2.0, 3.0]])
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(1)),
    )


def test_sqrt_transform_different_random_seeds() -> None:
    transformer = SqrtTensorTransformer()
    tensor = torch.tensor([[0.0, 4.0, 16.0], [1.0, 2.0, 3.0]])
    # the outputs must be equal because this transformer does not have randomness
    assert objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )
