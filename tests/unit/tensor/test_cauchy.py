from unittest.mock import Mock, patch

import torch
from pytest import mark, raises

from startorch.tensor import (
    Cauchy,
    RandCauchy,
    RandTruncCauchy,
    RandUniform,
    TruncCauchy,
)
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


############################
#     Tests for Cauchy     #
############################


def test_cauchy_str() -> None:
    assert str(
        Cauchy(loc=RandUniform(low=-1.0, high=1.0), scale=RandUniform(low=1.0, high=2.0))
    ).startswith("CauchyTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_cauchy_generate(size: tuple[int, ...]) -> None:
    tensor = Cauchy(
        loc=RandUniform(low=-1.0, high=1.0),
        scale=RandUniform(low=1.0, high=2.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_cauchy_generate_mock() -> None:
    generator = Cauchy(
        loc=RandUniform(low=-1.0, high=1.0),
        scale=RandUniform(low=1.0, high=2.0),
    )
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.cauchy.cauchy", mock):
        generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_cauchy_generate_same_random_seed() -> None:
    generator = Cauchy(loc=RandUniform(low=-1.0, high=1.0), scale=RandUniform(low=1.0, high=2.0))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_cauchy_generate_different_random_seeds() -> None:
    generator = Cauchy(loc=RandUniform(low=-1.0, high=1.0), scale=RandUniform(low=1.0, high=2.0))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


################################
#     Tests for RandCauchy     #
################################


def test_rand_cauchy_str() -> None:
    assert str(RandCauchy()).startswith("RandCauchyTensorGenerator(")


@mark.parametrize("loc", (-1.0, 0.0, 1.0))
def test_rand_cauchy_loc(loc: float) -> None:
    assert RandCauchy(loc=loc)._loc == loc


def test_rand_cauchy_loc_default() -> None:
    assert RandCauchy()._loc == 0.0


@mark.parametrize("scale", (1.0, 2.0))
def test_rand_cauchy_scale(scale: float) -> None:
    assert RandCauchy(scale=scale)._scale == scale


def test_rand_cauchy_scale_default() -> None:
    assert RandCauchy()._scale == 1.0


@mark.parametrize("scale", (0.0, -1.0))
def test_rand_cauchy_incorrect_scale(scale: float) -> None:
    with raises(ValueError, match="scale has to be greater than 0"):
        RandCauchy(scale=scale)


@mark.parametrize("size", SIZES)
def test_rand_cauchy_generate(size: tuple[int, ...]) -> None:
    tensor = RandCauchy().generate(size)
    assert tensor.data.shape == size
    assert tensor.data.dtype == torch.float


@mark.parametrize("loc", (0, 1))
@mark.parametrize("scale", (1, 2))
def test_rand_cauchy_generate_loc_scale(loc: float, scale: float) -> None:
    generator = RandCauchy(loc=loc, scale=scale)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.cauchy.rand_cauchy", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["loc"] == loc
        assert mock.call_args.kwargs["scale"] == scale


def test_rand_cauchy_generate_same_random_seed() -> None:
    generator = RandCauchy()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_cauchy_generate_different_random_seeds() -> None:
    generator = RandCauchy()
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


#####################################
#     Tests for RandTruncCauchy     #
#####################################


def test_rand_trunc_cauchy_str() -> None:
    assert str(RandTruncCauchy()).startswith("RandTruncCauchyTensorGenerator(")


@mark.parametrize("loc", (-1.0, 0.0, 1.0))
def test_rand_trunc_cauchy_loc(loc: float) -> None:
    assert RandTruncCauchy(loc=loc)._loc == loc


def test_rand_trunc_cauchy_loc_default() -> None:
    assert RandTruncCauchy()._loc == 0.0


@mark.parametrize("scale", (1.0, 2.0))
def test_rand_trunc_cauchy_scale(scale: float) -> None:
    assert RandTruncCauchy(scale=scale)._scale == scale


def test_rand_trunc_cauchy_scale_default() -> None:
    assert RandTruncCauchy()._scale == 1.0


@mark.parametrize("scale", (0.0, -1.0))
def test_rand_trunc_cauchy_incorrect_scale(scale: float) -> None:
    with raises(ValueError, match="scale has to be greater than 0"):
        RandTruncCauchy(scale=scale)


@mark.parametrize("min_value", (-1.0, -2.0))
def test_rand_trunc_cauchy_min_value(min_value: float) -> None:
    assert RandTruncCauchy(min_value=min_value)._min_value == min_value


def test_rand_trunc_cauchy_min_value_default() -> None:
    assert RandTruncCauchy()._min_value == -2.0


@mark.parametrize("max_value", (1.0, 2.0))
def test_rand_trunc_cauchy_max_value(max_value: float) -> None:
    assert RandTruncCauchy(max_value=max_value)._max_value == max_value


def test_rand_trunc_cauchy_max_value_default() -> None:
    assert RandTruncCauchy()._max_value == 2.0


def test_rand_trunc_cauchy_incorrect_min_max() -> None:
    with raises(ValueError, match="max_value (.*) has to be greater or equal to min_value"):
        RandTruncCauchy(min_value=1.0, max_value=-1.0)


@mark.parametrize("size", SIZES)
def test_rand_trunc_cauchy_generate(size: tuple[int, ...]) -> None:
    tensor = RandTruncCauchy().generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= -2.0
    assert tensor.max() < 2.0


@mark.parametrize("loc", (0, 1))
@mark.parametrize("scale", (1, 2))
def test_rand_trunc_cauchy_generate_loc_scale(loc: float, scale: float) -> None:
    generator = RandTruncCauchy(loc=loc, scale=scale)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.cauchy.rand_trunc_cauchy", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["loc"] == loc
        assert mock.call_args.kwargs["scale"] == scale


@mark.parametrize("max_value", (1.0, 2.0))
@mark.parametrize("min_value", (-1.0, -2.0))
def test_rand_trunc_cauchy_generate_max_min(max_value: float, min_value: float) -> None:
    generator = RandTruncCauchy(min_value=min_value, max_value=max_value)
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.cauchy.rand_trunc_cauchy", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        assert mock.call_args.kwargs["max_value"] == max_value
        assert mock.call_args.kwargs["min_value"] == min_value


def test_rand_trunc_cauchy_generate_same_random_seed() -> None:
    generator = RandTruncCauchy()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_rand_trunc_cauchy_generate_different_random_seeds() -> None:
    generator = RandTruncCauchy()
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


#################################
#     Tests for TruncCauchy     #
#################################


def test_trunc_cauchy_str() -> None:
    assert str(
        TruncCauchy(
            loc=RandUniform(low=-1.0, high=1.0),
            scale=RandUniform(low=1.0, high=2.0),
            min_value=RandUniform(low=-10.0, high=-5.0),
            max_value=RandUniform(low=5.0, high=10.0),
        )
    ).startswith("TruncCauchyTensorGenerator(")


@mark.parametrize("size", SIZES)
def test_trunc_cauchy_generate(size: tuple[int, ...]) -> None:
    tensor = TruncCauchy(
        loc=RandUniform(low=-1.0, high=1.0),
        scale=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    ).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float


def test_trunc_cauchy_generate_mock() -> None:
    generator = TruncCauchy(
        loc=RandUniform(low=-1.0, high=1.0),
        scale=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.cauchy.trunc_cauchy", mock):
        assert generator.generate(size=(2, 4)).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_trunc_cauchy_generate_same_random_seed() -> None:
    generator = TruncCauchy(
        loc=RandUniform(low=-1.0, high=1.0),
        scale=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_trunc_cauchy_generate_different_random_seeds() -> None:
    generator = TruncCauchy(
        loc=RandUniform(low=-1.0, high=1.0),
        scale=RandUniform(low=1.0, high=2.0),
        min_value=RandUniform(low=-10.0, high=-5.0),
        max_value=RandUniform(low=5.0, high=10.0),
    )
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
