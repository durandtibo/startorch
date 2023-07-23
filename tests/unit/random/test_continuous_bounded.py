import math
from unittest.mock import Mock, patch

import torch
from pytest import mark, raises

from startorch.random import (
    asinh_uniform,
    log_uniform,
    rand_asinh_uniform,
    rand_log_uniform,
    rand_trunc_cauchy,
    rand_trunc_exponential,
    rand_trunc_half_cauchy,
    rand_trunc_half_normal,
    rand_trunc_log_normal,
    rand_trunc_normal,
    rand_uniform,
    trunc_cauchy,
    trunc_exponential,
    trunc_half_cauchy,
    trunc_half_normal,
    trunc_log_normal,
    trunc_normal,
    uniform,
)
from startorch.utils.seed import get_torch_generator

TOLERANCE = 0.05


#######################################
#     Tests for rand_trunc_cauchy     #
#######################################


def test_rand_rand_trunc_cauchy_1d() -> None:
    values = rand_trunc_cauchy((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.min() >= -2.0
    assert values.max() <= 2.0


def test_rand_rand_trunc_cauchy_2d() -> None:
    values = rand_trunc_cauchy((1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.min() >= -2.0
    assert values.max() <= 2.0


@mark.parametrize("loc", (-1.0, 0.0, 1.0))
def test_rand_trunc_cauchy_loc(loc: float) -> None:
    values = rand_trunc_cauchy(
        (100000,),
        loc=loc,
        min_value=loc - 2.0,
        max_value=loc + 2.0,
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(loc), atol=TOLERANCE)
    assert values.min() >= loc - 2.0
    assert values.max() <= loc + 2.0


@mark.parametrize("scale", (0.1, 0.5, 1.0))
def test_rand_trunc_cauchy_scale(scale: float) -> None:
    values = rand_trunc_cauchy((100000,), scale=scale, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= -2.0
    assert values.max() <= 2.0


@mark.parametrize("scale", (0.0, -1.0))
def test_rand_trunc_cauchy_scale_incorrect(scale: float) -> None:
    with raises(ValueError):
        rand_trunc_cauchy((1000,), scale=scale, generator=get_torch_generator(1))


@mark.parametrize("min_value", (-2.0, -1.0))
@mark.parametrize("max_value", (2.0, 1.0))
def test_rand_trunc_cauchy_min_max(min_value: float, max_value: float) -> None:
    values = rand_trunc_cauchy(
        (100000,), min_value=min_value, max_value=max_value, generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= min_value
    assert values.max() <= max_value


def test_rand_trunc_cauchy_incorrect_min_max() -> None:
    with raises(
        ValueError, match="`max_value` (.*) has to be greater or equal to `min_value` (.*)"
    ):
        rand_trunc_cauchy((1000,), min_value=1, max_value=-1, generator=get_torch_generator(1))


def test_rand_trunc_cauchy_same_random_seed() -> None:
    assert rand_trunc_cauchy((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_cauchy((1000,), generator=get_torch_generator(1))
    )


def test_rand_trunc_cauchy_different_random_seeds() -> None:
    assert not rand_trunc_cauchy((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_cauchy((1000,), generator=get_torch_generator(2))
    )


##################################
#     Tests for trunc_cauchy     #
##################################


def test_rand_trunc_cauchy_1d() -> None:
    values = trunc_cauchy(
        loc=torch.zeros(100000),
        scale=torch.ones(100000),
        min_value=torch.full((100000,), -2.0),
        max_value=torch.full((100000,), 2.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.min() >= -2.0
    assert values.max() <= 2.0


def test_rand_trunc_cauchy_2d() -> None:
    values = trunc_cauchy(
        loc=torch.zeros(1000, 100),
        scale=torch.ones(1000, 100),
        min_value=torch.full((1000, 100), -2.0),
        max_value=torch.full((1000, 100), 2.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.min() >= -2.0
    assert values.max() <= 2.0


@mark.parametrize("loc", (-1.0, 0.0, 1.0))
def test_trunc_cauchy_loc(loc: float) -> None:
    values = trunc_cauchy(
        loc=torch.full((100000,), loc),
        scale=torch.ones(100000),
        min_value=torch.full((100000,), loc - 2.0),
        max_value=torch.full((100000,), loc + 2.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(loc), atol=TOLERANCE)
    assert values.min() >= loc - 2.0
    assert values.max() <= loc + 2.0


@mark.parametrize("scale", (0.1, 0.5, 1.0))
def test_trunc_cauchy_scale(scale: float) -> None:
    values = trunc_cauchy(
        loc=torch.zeros(100000),
        scale=torch.full((100000,), scale),
        min_value=torch.full((100000,), -2.0),
        max_value=torch.full((100000,), 2.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= -2.0
    assert values.max() <= 2.0


@mark.parametrize("scale", (0.0, -1.0))
def test_trunc_cauchy_scale_incorrect(scale: float) -> None:
    with raises(ValueError):
        trunc_cauchy(
            loc=torch.zeros(100000),
            scale=torch.full((100000,), scale),
            min_value=torch.full((100000,), -2.0),
            max_value=torch.full((100000,), 2.0),
            generator=get_torch_generator(1),
        )


@mark.parametrize("min_value", (-2.0, -1.0))
@mark.parametrize("max_value", (2.0, 1.0))
def test_trunc_cauchy_min_max(min_value: float, max_value: float) -> None:
    values = trunc_cauchy(
        loc=torch.zeros(100000),
        scale=torch.ones(100000),
        min_value=torch.full((100000,), min_value),
        max_value=torch.full((100000,), max_value),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= min_value
    assert values.max() <= max_value


def test_trunc_cauchy_incorrect_min_max() -> None:
    with raises(
        ValueError,
        match=(
            "Found at least one value in `min_value` that is higher than "
            "its associated `max_value`"
        ),
    ):
        trunc_cauchy(
            loc=torch.zeros(1000),
            scale=torch.ones(1000),
            min_value=torch.full((1000,), 1.0),
            max_value=torch.full((1000,), -1.0),
            generator=get_torch_generator(1),
        )


def test_trunc_cauchy_shape_mismatch_loc() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_cauchy(
            loc=torch.zeros(5),
            scale=torch.ones(6),
            min_value=torch.full((6,), -3.0),
            max_value=torch.full((6,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_cauchy_shape_mismatch_scale() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_cauchy(
            loc=torch.zeros(6),
            scale=torch.ones(5),
            min_value=torch.full((6,), -3.0),
            max_value=torch.full((6,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_cauchy_shape_mismatch_min_value() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_cauchy(
            loc=torch.zeros(6),
            scale=torch.ones(6),
            min_value=torch.full((5,), -3.0),
            max_value=torch.full((6,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_cauchy_shape_mismatch_max_value() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_cauchy(
            loc=torch.zeros(6),
            scale=torch.ones(6),
            min_value=torch.full((6,), -3.0),
            max_value=torch.full((5,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_cauchy_same_random_seed() -> None:
    assert trunc_cauchy(
        loc=torch.zeros(1000),
        scale=torch.ones(1000),
        min_value=torch.full((1000,), -3.0),
        max_value=torch.full((1000,), 3.0),
        generator=get_torch_generator(1),
    ).equal(
        trunc_cauchy(
            loc=torch.zeros(1000),
            scale=torch.ones(1000),
            min_value=torch.full((1000,), -3.0),
            max_value=torch.full((1000,), 3.0),
            generator=get_torch_generator(1),
        )
    )


def test_trunc_cauchy_different_random_seeds() -> None:
    assert not trunc_cauchy(
        loc=torch.zeros(1000),
        scale=torch.ones(1000),
        min_value=torch.full((1000,), -3.0),
        max_value=torch.full((1000,), 3.0),
        generator=get_torch_generator(1),
    ).equal(
        trunc_cauchy(
            loc=torch.zeros(1000),
            scale=torch.ones(1000),
            min_value=torch.full((1000,), -3.0),
            max_value=torch.full((1000,), 3.0),
            generator=get_torch_generator(2),
        )
    )


############################################
#     Tests for rand_trunc_exponential     #
############################################


def test_rand_trunc_exponential_1d() -> None:
    values = rand_trunc_exponential((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(2.0).log(), atol=TOLERANCE)
    assert values.min() >= 0.0
    assert values.max() <= 5.0


def test_rand_trunc_exponential_2d() -> None:
    values = rand_trunc_exponential((1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(2.0).log(), atol=TOLERANCE)
    assert values.min() >= 0.0
    assert values.max() <= 5.0


@mark.parametrize("rate", (0.5, 2.0))
def test_rand_trunc_exponential_rate(rate: float) -> None:
    values = rand_trunc_exponential(
        (100000,), rate=rate, max_value=10, generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(2.0).log().div(rate), atol=TOLERANCE)
    assert values.min() >= 0.0
    assert values.max() <= 10.0


@mark.parametrize("rate", (0.0, -1.0))
def test_rand_trunc_exponential_incorrect_rate(rate: float) -> None:
    with raises(ValueError):
        rand_trunc_exponential((1000,), rate=rate, generator=get_torch_generator(1))


@mark.parametrize("max_value", (2.0, 1.0))
def test_rand_trunc_exponential_max_value(max_value: float) -> None:
    values = rand_trunc_exponential(
        (100000,), max_value=max_value, generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= max_value


@mark.parametrize("max_value", (0.0, -1.0))
def test_rand_trunc_exponential_incorrect_max_value(max_value: float) -> None:
    with raises(ValueError, match="`max_value` has to be greater than 0"):
        rand_trunc_exponential((1000,), max_value=max_value, generator=get_torch_generator(1))


def test_rand_trunc_exponential_same_random_seed() -> None:
    assert rand_trunc_exponential((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_exponential((1000,), generator=get_torch_generator(1))
    )


def test_rand_trunc_exponential_different_random_seeds() -> None:
    assert not rand_trunc_exponential((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_exponential((1000,), generator=get_torch_generator(2))
    )


#######################################
#     Tests for trunc_exponential     #
#######################################


def test_trunc_exponential_1d() -> None:
    values = trunc_exponential(
        rate=torch.ones(100000),
        max_value=torch.full((100000,), 3.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(2.0).log(), atol=TOLERANCE)
    assert values.min() >= 0.0
    assert values.max() <= 5.0


def test_trunc_exponential_2d() -> None:
    values = trunc_exponential(
        rate=torch.ones(1000, 100),
        max_value=torch.full(
            (1000, 100),
            3.0,
        ),
        generator=get_torch_generator(1),
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(2.0).log(), atol=TOLERANCE)
    assert values.min() >= 0.0
    assert values.max() <= 5.0


@mark.parametrize("rate", (0.5, 2.0))
def test_trunc_exponential_rate(rate: float) -> None:
    values = trunc_exponential(
        rate=torch.full((100000,), rate),
        max_value=torch.full((100000,), 10.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.median().allclose(torch.tensor(2.0).log().div(rate), atol=TOLERANCE)
    assert values.min() >= 0.0
    assert values.max() <= 10.0


@mark.parametrize("rate", (0.0, -1.0))
def test_trunc_exponential_incorrect_rate(rate: float) -> None:
    with raises(ValueError, match="All the `rate` values have to be greater than 0"):
        trunc_exponential(
            rate=torch.full((100000,), rate),
            max_value=torch.full((100000,), 3.0),
            generator=get_torch_generator(1),
        )


@mark.parametrize("max_value", (2.0, 1.0))
def test_trunc_exponential_max_value(max_value: float) -> None:
    values = trunc_exponential(
        rate=torch.ones(100000),
        max_value=torch.full((100000,), max_value),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= max_value


@mark.parametrize("max_value", (0.0, -1.0))
def test_trunc_exponential_incorrect_max_value(max_value: float) -> None:
    with raises(
        ValueError, match="Found at least one value in `max_value` that is lower or equal to 0"
    ):
        trunc_exponential(
            rate=torch.ones(100000),
            max_value=torch.full((100000,), max_value),
            generator=get_torch_generator(1),
        )


def test_trunc_exponential_shape_mismatch() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_exponential(
            rate=torch.ones(5),
            max_value=torch.full((6,), 5.0),
            generator=get_torch_generator(1),
        )


def test_trunc_exponential_same_random_seed() -> None:
    assert trunc_exponential(
        rate=torch.ones(1000), max_value=torch.full((1000,), 3.0), generator=get_torch_generator(1)
    ).equal(
        trunc_exponential(
            rate=torch.ones(1000),
            max_value=torch.full((1000,), 3.0),
            generator=get_torch_generator(1),
        )
    )


def test_trunc_exponential_different_random_seeds() -> None:
    assert not trunc_exponential(
        rate=torch.ones(1000), max_value=torch.full((1000,), 3.0), generator=get_torch_generator(1)
    ).equal(
        trunc_exponential(
            rate=torch.ones(1000),
            max_value=torch.full((1000,), 3.0),
            generator=get_torch_generator(2),
        )
    )


############################################
#     Tests for rand_trunc_half_cauchy     #
############################################


def test_rand_trunc_half_cauchy_1d() -> None:
    values = rand_trunc_half_cauchy((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= 4.0


def test_rand_trunc_half_cauchy_2d() -> None:
    values = rand_trunc_half_cauchy((1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= 4.0


def test_rand_trunc_half_cauchy_different_scale() -> None:
    assert not rand_trunc_half_cauchy((100000,), scale=1.0, generator=get_torch_generator(1)).equal(
        rand_trunc_half_cauchy((100000,), scale=0.1, generator=get_torch_generator(1))
    )


@mark.parametrize("scale", (0.0, -1.0))
def test_rand_trunc_half_cauchy_incorrect_scale(scale: float) -> None:
    with raises(ValueError):
        rand_trunc_half_cauchy((1000,), scale=scale, generator=get_torch_generator(1))


@mark.parametrize("max_value", (2.0, 1.0))
def test_rand_trunc_half_cauchy_max_value(max_value: float) -> None:
    values = rand_trunc_half_cauchy(
        (100000,), max_value=max_value, generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= max_value


@mark.parametrize("max_value", (0.0, -1.0))
def test_rand_trunc_half_cauchy_incorrect_max_value(max_value: float) -> None:
    with raises(ValueError, match="`max_value` has to be greater than 0"):
        rand_trunc_half_cauchy((1000,), max_value=max_value, generator=get_torch_generator(1))


def test_rand_trunc_half_cauchy_same_random_seed() -> None:
    assert rand_trunc_half_cauchy((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_half_cauchy((1000,), generator=get_torch_generator(1))
    )


def test_rand_trunc_half_cauchy_different_random_seeds() -> None:
    assert not rand_trunc_half_cauchy((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_half_cauchy((1000,), generator=get_torch_generator(2))
    )


#######################################
#     Tests for trunc_half_cauchy     #
#######################################


def test_trunc_half_cauchy_1d() -> None:
    values = trunc_half_cauchy(
        torch.ones(100000), torch.full((100000,), 4.0), generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= 4.0


def test_trunc_half_cauchy_2d() -> None:
    values = trunc_half_cauchy(
        torch.ones(1000, 100),
        torch.full(
            (1000, 100),
            4.0,
        ),
        generator=get_torch_generator(1),
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= 4.0


def test_trunc_half_cauchy_different_scale() -> None:
    assert not trunc_half_cauchy(
        torch.ones(100000), torch.full((100000,), 4.0), generator=get_torch_generator(1)
    ).equal(
        trunc_half_cauchy(
            torch.full((100000,), 0.1), torch.full((100000,), 4.0), generator=get_torch_generator(1)
        )
    )


@mark.parametrize("scale", (0.0, -1.0))
def test_trunc_half_cauchy_incorrect_scale(scale: float) -> None:
    with raises(ValueError, match="All the `scale` values have to be greater than 0"):
        trunc_half_cauchy(
            torch.full((1000,), scale), torch.full((1000,), 4.0), generator=get_torch_generator(1)
        )


@mark.parametrize("max_value", (2.0, 1.0))
def test_trunc_half_cauchy_max_value(max_value: float) -> None:
    values = trunc_half_cauchy(
        torch.ones(100000), torch.full((100000,), max_value), generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= max_value


@mark.parametrize("max_value", (0.0, -1.0))
def test_trunc_half_cauchy_max_value_incorrect(max_value: float) -> None:
    with raises(
        ValueError, match="Found at least one value in `max_value` that is lower or equal to 0"
    ):
        trunc_half_cauchy(
            torch.ones(100000), torch.full((100000,), max_value), generator=get_torch_generator(1)
        )


def test_trunc_half_cauchy_shape_mismatch() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_half_cauchy(
            scale=torch.ones(5),
            max_value=torch.full((6,), 5.0),
            generator=get_torch_generator(1),
        )


def test_trunc_half_cauchy_same_random_seed() -> None:
    assert trunc_half_cauchy(
        torch.ones(1000), torch.full((1000,), 3.0), generator=get_torch_generator(1)
    ).equal(
        trunc_half_cauchy(
            torch.ones(1000), torch.full((1000,), 3.0), generator=get_torch_generator(1)
        )
    )


def test_trunc_half_cauchy_different_random_seeds() -> None:
    assert not trunc_half_cauchy(
        torch.ones(1000), torch.full((1000,), 3.0), generator=get_torch_generator(1)
    ).equal(
        trunc_half_cauchy(
            torch.ones(1000), torch.full((1000,), 3.0), generator=get_torch_generator(2)
        )
    )


############################################
#     Tests for rand_trunc_half_normal     #
############################################


def test_rand_trunc_half_normal_1d() -> None:
    values = rand_trunc_half_normal((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(math.sqrt(2 / math.pi)), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(math.sqrt(1 - 2 / math.pi)), atol=TOLERANCE)
    assert values.min() >= 0.0
    assert values.max() <= 5.0


def test_rand_trunc_half_normal_2d() -> None:
    values = rand_trunc_half_normal((1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(math.sqrt(2 / math.pi)), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(math.sqrt(1 - 2 / math.pi)), atol=TOLERANCE)
    assert values.min() >= 0.0
    assert values.max() <= 5.0


@mark.parametrize("std", (0.1, 0.5, 1.0))
def test_rand_trunc_half_normal_std(std: float) -> None:
    values = rand_trunc_half_normal((100000,), std=std, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(std * math.sqrt(2 / math.pi)), atol=TOLERANCE)
    assert values.std(dim=None).allclose(
        torch.tensor(std * math.sqrt(1 - 2 / math.pi)), atol=TOLERANCE
    )
    assert values.min() >= 0.0
    assert values.max() <= 5.0


def test_rand_trunc_half_normal_incorrect_std() -> None:
    with raises(ValueError):
        rand_trunc_half_normal((1000,), std=-1, generator=get_torch_generator(1))


@mark.parametrize("max_value", (2.0, 1.0))
def test_rand_trunc_half_normal_max_value(max_value: float) -> None:
    values = rand_trunc_half_normal(
        (100000,), max_value=max_value, generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= max_value


def test_rand_trunc_half_normal_incorrect_max_value() -> None:
    with raises(ValueError, match="`max_value` has to be greater than 0"):
        rand_trunc_half_normal((1000,), max_value=0, generator=get_torch_generator(1))


def test_rand_trunc_half_normal_same_random_seed() -> None:
    assert rand_trunc_half_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_half_normal((1000,), generator=get_torch_generator(1))
    )


def test_rand_trunc_half_normal_different_random_seeds() -> None:
    assert not rand_trunc_half_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_half_normal((1000,), generator=get_torch_generator(2))
    )


#######################################
#     Tests for trunc_half_normal     #
#######################################


def test_trunc_half_normal_1d() -> None:
    values = trunc_half_normal(
        std=torch.ones(100000),
        max_value=torch.full((100000,), 100.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(math.sqrt(2 / math.pi)), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(math.sqrt(1 - 2 / math.pi)), atol=TOLERANCE)
    assert values.min() >= 0.0
    assert values.max() <= 100.0


def test_trunc_half_normal_2d() -> None:
    values = trunc_half_normal(
        std=torch.ones(1000, 100),
        max_value=torch.full((1000, 100), 100.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(math.sqrt(2 / math.pi)), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(math.sqrt(1 - 2 / math.pi)), atol=TOLERANCE)
    assert values.min() >= 0.0
    assert values.max() <= 100.0


@mark.parametrize("std", (0.1, 0.5, 1.0))
def test_trunc_half_normal_std(std: float) -> None:
    values = trunc_half_normal(
        std=torch.full((1000000,), std),
        max_value=torch.full((1000000,), 100.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (1000000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(std * math.sqrt(2 / math.pi)), atol=TOLERANCE)
    assert values.std(dim=None).allclose(
        torch.tensor(std * math.sqrt(1 - 2 / math.pi)), atol=TOLERANCE
    )
    assert values.min() >= 0.0
    assert values.max() <= 100.0


@mark.parametrize("std", (0.0, -1.0))
def test_trunc_half_normal_incorrect_std(std: float) -> None:
    with raises(ValueError, match="All the `std` values have to be greater than 0"):
        trunc_half_normal(
            std=torch.full((1000,), std),
            max_value=torch.full((1000,), 100.0),
            generator=get_torch_generator(1),
        )


@mark.parametrize("max_value", (2.0, 1.0))
def test_trunc_half_normal_max_value(max_value: float) -> None:
    values = trunc_half_normal(
        torch.ones(100000),
        max_value=torch.full((100000,), max_value),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= max_value


@mark.parametrize("max_value", (0.0, -1.0))
def test_trunc_half_normal_incorrect_max_value(max_value: float) -> None:
    with raises(ValueError, match="All the `max_value` values must be greater than 0"):
        trunc_half_normal(
            std=torch.ones(1000),
            max_value=torch.full((1000,), max_value),
            generator=get_torch_generator(1),
        )


def test_trunc_half_normal_shape_mismatch() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_half_normal(
            std=torch.ones(5),
            max_value=torch.full((6,), 5.0),
            generator=get_torch_generator(1),
        )


def test_trunc_half_normal_same_random_seed() -> None:
    assert trunc_half_normal(
        std=torch.ones(1000), max_value=torch.full((1000,), 5.0), generator=get_torch_generator(1)
    ).equal(
        trunc_half_normal(
            std=torch.ones(1000),
            max_value=torch.full((1000,), 5.0),
            generator=get_torch_generator(1),
        )
    )


def test_trunc_half_normal_different_random_seeds() -> None:
    assert not trunc_half_normal(
        std=torch.ones(1000), max_value=torch.full((1000,), 5.0), generator=get_torch_generator(1)
    ).equal(
        trunc_half_normal(
            std=torch.ones(1000),
            max_value=torch.full((1000,), 5.0),
            generator=get_torch_generator(2),
        )
    )


###########################################
#     Tests for rand_trunc_log_normal     #
###########################################


def test_rand_trunc_log_normal_1d() -> None:
    values = rand_trunc_log_normal(
        (100000,), min_value=0.0, max_value=100.0, generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.5).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0), rtol=TOLERANCE)
    assert values.std(dim=None).allclose(
        torch.tensor(1.0).exp().sub(1.0).mul(torch.tensor(1.0).exp()).sqrt(), rtol=TOLERANCE
    )
    assert values.min() >= 0.0
    assert values.max() <= 100.0


def test_rand_trunc_log_normal_2d() -> None:
    values = rand_trunc_log_normal(
        (1000, 100), min_value=0.0, max_value=100.0, generator=get_torch_generator(1)
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.5).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0), rtol=TOLERANCE)
    assert values.std(dim=None).allclose(
        torch.tensor(1.0).exp().sub(1.0).mul(torch.tensor(1.0).exp()).sqrt(), rtol=TOLERANCE
    )
    assert values.min() >= 0.0
    assert values.max() <= 100.0


@mark.parametrize("mean", (-1.0, 0.0, 1.0))
def test_rand_trunc_log_normal_mean(mean: float) -> None:
    values = rand_trunc_log_normal(
        (100000,),
        mean=mean,
        min_value=0.0,
        max_value=1000.0,
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(mean + 0.5).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(mean).exp(), rtol=TOLERANCE)
    assert values.std(dim=None).allclose(
        torch.tensor(1.0).exp().sub(1.0).mul(torch.tensor(2.0 * mean + 1.0).exp()).sqrt(),
        rtol=TOLERANCE,
    )
    assert values.min() >= 0.0
    assert values.max() <= 1000.0


@mark.parametrize("std", (0.25, 0.5, 1.0))
def test_rand_trunc_log_normal_std(std: float) -> None:
    values = rand_trunc_log_normal(
        (100000,), std=std, min_value=0.0, max_value=10000.0, generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.5 * std**2).exp(), rtol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0), rtol=TOLERANCE)
    assert values.std(dim=None).allclose(
        torch.tensor(std**2).exp().sub(1.0).mul(torch.tensor(std**2).exp()).sqrt(),
        rtol=TOLERANCE,
    )
    assert values.min() >= 0.0
    assert values.max() <= 10000.0


@mark.parametrize("std", (0.0, -1.0))
def test_rand_trunc_log_normal_incorrect_std(std: float) -> None:
    with raises(ValueError):
        rand_trunc_log_normal((1000,), std=std, generator=get_torch_generator(1))


@mark.parametrize("min_value", (0.0, 1.0))
@mark.parametrize("max_value", (2.0, 3.0))
def test_rand_trunc_log_normal_min_max(min_value: float, max_value: float) -> None:
    values = rand_trunc_log_normal(
        (1000,), min_value=min_value, max_value=max_value, generator=get_torch_generator(1)
    )
    assert values.shape == (1000,)
    assert values.dtype == torch.float
    assert values.min() >= min_value
    assert values.max() <= max_value


def test_rand_trunc_log_normal_min_max_default() -> None:
    values = rand_trunc_log_normal((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() <= 5.0


def test_rand_trunc_log_normal_incorrect_min_max() -> None:
    with raises(
        ValueError, match="`max_value` (.*) has to be greater or equal to `min_value` (.*)"
    ):
        rand_trunc_log_normal((1000,), min_value=1, max_value=-1, generator=get_torch_generator(1))


def test_rand_trunc_log_normal_same_random_seed() -> None:
    assert rand_trunc_log_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_log_normal((1000,), generator=get_torch_generator(1))
    )


def test_rand_trunc_log_normal_different_random_seeds() -> None:
    assert not rand_trunc_log_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_log_normal((1000,), generator=get_torch_generator(2))
    )


######################################
#     Tests for trunc_log_normal     #
######################################


def test_trunc_log_normal_1d() -> None:
    values = trunc_log_normal(
        mean=torch.zeros(100000),
        std=torch.ones(100000),
        min_value=torch.zeros(100000),
        max_value=torch.full((100000,), 100.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.5).exp(), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0), rtol=TOLERANCE)
    assert values.std(dim=None).allclose(
        torch.tensor(1.0).exp().sub(1.0).mul(torch.tensor(1.0).exp()).sqrt(), atol=TOLERANCE
    )
    assert values.min() >= 0.0
    assert values.max() <= 100.0


def test_trunc_log_normal_2d() -> None:
    values = trunc_log_normal(
        mean=torch.zeros(1000, 100),
        std=torch.ones(1000, 100),
        min_value=torch.zeros(1000, 100),
        max_value=torch.full((1000, 100), 100.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.5).exp(), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0), rtol=TOLERANCE)
    assert values.std(dim=None).allclose(
        torch.tensor(1.0).exp().sub(1.0).mul(torch.tensor(1.0).exp()).sqrt(), atol=TOLERANCE
    )
    assert values.min() >= 0.0
    assert values.max() <= 100.0


@mark.parametrize("mean", (-1.0, 0.0, 1.0))
def test_trunc_log_normal_mean(mean: float) -> None:
    values = trunc_log_normal(
        mean=torch.full((100000,), mean),
        std=torch.ones(100000),
        min_value=torch.zeros(100000),
        max_value=torch.full((100000,), 1000.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(mean + 0.5).exp(), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(mean).exp(), rtol=TOLERANCE)
    assert values.std(dim=None).allclose(
        torch.tensor(1.0).exp().sub(1.0).mul(torch.tensor(2.0 * mean + 1.0).exp()).sqrt(),
        atol=TOLERANCE,
    )
    assert values.min() >= 0.0
    assert values.max() <= 1000.0


@mark.parametrize("std", (0.25, 0.5, 1.0))
def test_trunc_log_normal_std(std: float) -> None:
    values = trunc_log_normal(
        mean=torch.zeros(100000),
        std=torch.full((100000,), std),
        min_value=torch.zeros(100000),
        max_value=torch.full((100000,), 1000.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.5 * std**2).exp(), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(1.0), rtol=TOLERANCE)
    assert values.std(dim=None).allclose(
        torch.tensor(std**2).exp().sub(1.0).mul(torch.tensor(std**2).exp()).sqrt(),
        atol=TOLERANCE,
    )
    assert values.min() >= 0.0
    assert values.max() <= 10000.0


@mark.parametrize("std", (0.0, -1.0))
def test_trunc_log_normal_std_incorrect(std: float) -> None:
    with raises(ValueError):
        trunc_log_normal(
            mean=torch.zeros(1000),
            std=torch.full((1000,), std),
            min_value=torch.zeros(1000),
            max_value=torch.full((1000,), 100.0),
            generator=get_torch_generator(1),
        )


@mark.parametrize("min_value", (0.0, 1.0))
@mark.parametrize("max_value", (2.0, 3.0))
def test_trunc_log_normal_min_max(min_value: float, max_value: float) -> None:
    values = trunc_log_normal(
        mean=torch.zeros(100000),
        std=torch.ones(100000),
        min_value=torch.full((100000,), min_value),
        max_value=torch.full((100000,), max_value),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= min_value
    assert values.max() <= max_value


def test_trunc_log_normal_incorrect_min_max() -> None:
    with raises(
        ValueError,
        match=(
            "Found at least one value in `min_value` that is higher than its associated "
            "`max_value`"
        ),
    ):
        trunc_log_normal(
            mean=torch.zeros(10),
            std=torch.ones(10),
            min_value=torch.full((10,), 5.0),
            max_value=torch.full((10,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_log_normal_shape_mismatch_mean() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_log_normal(
            mean=torch.zeros(5),
            std=torch.ones(6),
            min_value=torch.zeros(6),
            max_value=torch.full((6,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_log_normal_shape_mismatch_std() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_log_normal(
            mean=torch.zeros(6),
            std=torch.ones(5),
            min_value=torch.zeros(6),
            max_value=torch.full((6,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_log_normal_shape_mismatch_min_value() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_log_normal(
            mean=torch.zeros(6),
            std=torch.ones(6),
            min_value=torch.zeros(5),
            max_value=torch.full((6,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_log_normal_shape_mismatch_max_value() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_log_normal(
            mean=torch.zeros(6),
            std=torch.ones(6),
            min_value=torch.zeros(6),
            max_value=torch.full((5,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_log_normal_same_random_seed() -> None:
    assert trunc_log_normal(
        mean=torch.zeros(1000),
        std=torch.ones(1000),
        min_value=torch.zeros(1000),
        max_value=torch.full((1000,), 10.0),
        generator=get_torch_generator(1),
    ).equal(
        trunc_log_normal(
            mean=torch.zeros(1000),
            std=torch.ones(1000),
            min_value=torch.zeros(1000),
            max_value=torch.full((1000,), 10.0),
            generator=get_torch_generator(1),
        )
    )


def test_trunc_log_normal_different_random_seeds() -> None:
    assert not trunc_log_normal(
        mean=torch.zeros(1000),
        std=torch.ones(1000),
        min_value=torch.zeros(1000),
        max_value=torch.full((1000,), 10.0),
        generator=get_torch_generator(1),
    ).equal(
        trunc_log_normal(
            mean=torch.zeros(1000),
            std=torch.ones(1000),
            min_value=torch.zeros(1000),
            max_value=torch.full((1000,), 10.0),
            generator=get_torch_generator(2),
        )
    )


#######################################
#     Tests for rand_trunc_normal     #
#######################################


def test_rand_trunc_normal_1d() -> None:
    values = rand_trunc_normal(
        (100000,), min_value=-100, max_value=100, generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.min() >= -100.0
    assert values.max() <= 100.0


def test_rand_trunc_normal_2d() -> None:
    values = rand_trunc_normal(
        (1000, 100), min_value=-100, max_value=100, generator=get_torch_generator(1)
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.min() >= -100.0
    assert values.max() <= 100.0


@mark.parametrize("mean", (-1.0, 0.0, 1.0))
def test_rand_trunc_normal_mean(mean: float) -> None:
    values = rand_trunc_normal(
        (100000,),
        mean=mean,
        min_value=mean - 100,
        max_value=mean + 100,
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(mean), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.min() >= mean - 100.0
    assert values.max() <= mean + 100.0


@mark.parametrize("std", (1.0, 2.0, 5.0))
def test_rand_trunc_normal_std(std: float) -> None:
    values = rand_trunc_normal(
        (100000,), std=std, min_value=-100, max_value=100, generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(std), atol=TOLERANCE)
    assert values.min() >= -100.0
    assert values.max() <= 100.0


def test_rand_trunc_normal_incorrect_std() -> None:
    with raises(ValueError):
        rand_trunc_normal((1000,), std=-1, generator=get_torch_generator(1))


@mark.parametrize("min_value", (-2.0, -1.0))
@mark.parametrize("max_value", (2.0, 1.0))
def test_rand_trunc_normal_min_max(min_value: float, max_value: float) -> None:
    values = rand_trunc_normal(
        (1000,), min_value=min_value, max_value=max_value, generator=get_torch_generator(1)
    )
    assert values.shape == (1000,)
    assert values.dtype == torch.float
    assert values.min() >= min_value
    assert values.max() <= max_value


def test_rand_trunc_normal_min_max_default() -> None:
    values = rand_trunc_normal((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= -3.0
    assert values.max() <= 3.0


def test_rand_trunc_normal_incorrect_min_max() -> None:
    with raises(
        ValueError, match="`max_value` (.*) has to be greater or equal to `min_value` (.*)"
    ):
        rand_trunc_normal((1000,), min_value=1, max_value=-1, generator=get_torch_generator(1))


def test_rand_trunc_normal_same_random_seed() -> None:
    assert rand_trunc_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_normal((1000,), generator=get_torch_generator(1))
    )


def test_rand_trunc_normal_different_random_seeds() -> None:
    assert not rand_trunc_normal((1000,), generator=get_torch_generator(1)).equal(
        rand_trunc_normal((1000,), generator=get_torch_generator(2))
    )


##################################
#     Tests for trunc_normal     #
##################################


def test_trunc_normal_1d() -> None:
    values = trunc_normal(
        mean=torch.zeros(100000),
        std=torch.ones(100000),
        min_value=torch.full((100000,), -100.0),
        max_value=torch.full((100000,), 100.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.min() >= -100.0
    assert values.max() <= 100.0


def test_trunc_normal_2d() -> None:
    values = trunc_normal(
        mean=torch.zeros(1000, 100),
        std=torch.ones(1000, 100),
        min_value=torch.full((1000, 100), -100.0),
        max_value=torch.full((1000, 100), 100.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.min() >= -100.0
    assert values.max() <= 100.0


@mark.parametrize("mean", (-1.0, 0.0, 1.0))
def test_trunc_normal_mean(mean: float) -> None:
    values = trunc_normal(
        mean=torch.full((100000,), mean),
        std=torch.ones(100000),
        min_value=torch.full((100000,), mean - 100.0),
        max_value=torch.full((100000,), mean + 100.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(mean), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(1.0), atol=TOLERANCE)
    assert values.min() >= mean - 100.0
    assert values.max() <= mean + 100.0


@mark.parametrize("std", (1.0, 2.0, 5.0))
def test_trunc_normal_std(std: float) -> None:
    values = trunc_normal(
        mean=torch.zeros(100000),
        std=torch.full((100000,), std),
        min_value=torch.full((100000,), -100.0),
        max_value=torch.full((100000,), 100.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.std(dim=None).allclose(torch.tensor(std), atol=TOLERANCE)
    assert values.min() >= -100.0
    assert values.max() <= 100.0


@mark.parametrize("std", (0.0, -1.0))
def test_trunc_normal_std_incorrect(std: float) -> None:
    with raises(ValueError):
        trunc_normal(
            mean=torch.zeros(1000),
            std=torch.full((1000,), std),
            min_value=torch.full((1000,), -100.0),
            max_value=torch.full((1000,), 100.0),
            generator=get_torch_generator(1),
        )


@mark.parametrize("min_value", (-2.0, -1.0))
@mark.parametrize("max_value", (2.0, 1.0))
def test_trunc_normal_min_max(min_value: float, max_value: float) -> None:
    values = trunc_normal(
        mean=torch.zeros(100000),
        std=torch.ones(100000),
        min_value=torch.full((100000,), min_value),
        max_value=torch.full((100000,), max_value),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= min_value
    assert values.max() <= max_value


def test_trunc_normal_incorrect_min_max() -> None:
    with raises(ValueError):
        trunc_normal(
            mean=torch.zeros(1000),
            std=torch.ones(1000),
            min_value=torch.full((1000,), 1.0),
            max_value=torch.full((1000,), -1.0),
            generator=get_torch_generator(1),
        )


def test_trunc_normal_shape_mismatch_mean() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_normal(
            mean=torch.zeros(5),
            std=torch.ones(6),
            min_value=torch.full((6,), -3.0),
            max_value=torch.full((6,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_normal_shape_mismatch_std() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_normal(
            mean=torch.zeros(6),
            std=torch.ones(5),
            min_value=torch.full((6,), -3.0),
            max_value=torch.full((6,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_normal_shape_mismatch_min_value() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_normal(
            mean=torch.zeros(6),
            std=torch.ones(6),
            min_value=torch.full((5,), -3.0),
            max_value=torch.full((6,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_normal_shape_mismatch_max_value() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        trunc_normal(
            mean=torch.zeros(6),
            std=torch.ones(6),
            min_value=torch.full((6,), -3.0),
            max_value=torch.full((5,), 3.0),
            generator=get_torch_generator(1),
        )


def test_trunc_normal_same_random_seed() -> None:
    assert trunc_normal(
        mean=torch.zeros(1000),
        std=torch.ones(1000),
        min_value=torch.full((1000,), -100.0),
        max_value=torch.full((1000,), 100.0),
        generator=get_torch_generator(1),
    ).equal(
        trunc_normal(
            mean=torch.zeros(1000),
            std=torch.ones(1000),
            min_value=torch.full((1000,), -100.0),
            max_value=torch.full((1000,), 100.0),
            generator=get_torch_generator(1),
        )
    )


def test_trunc_normal_different_random_seeds() -> None:
    assert not trunc_normal(
        mean=torch.zeros(1000),
        std=torch.ones(1000),
        min_value=torch.full((1000,), -100.0),
        max_value=torch.full((1000,), 100.0),
        generator=get_torch_generator(1),
    ).equal(
        trunc_normal(
            mean=torch.zeros(1000),
            std=torch.ones(1000),
            min_value=torch.full((1000,), -100.0),
            max_value=torch.full((1000,), 100.0),
            generator=get_torch_generator(2),
        )
    )


##################################
#     Tests for rand_uniform     #
##################################


def test_rand_uniform_1d() -> None:
    values = rand_uniform((100000,), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() < 1.0


def test_rand_uniform_2d() -> None:
    values = rand_uniform((1000, 100), generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() < 1.0


@mark.parametrize("low", (-1.0, 0.0, 1.0))
def test_rand_uniform_low(low: float) -> None:
    values = rand_uniform((100000,), low=low, high=2.0, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= low
    assert values.max() < 2.0


@mark.parametrize("high", (0.1, 0.5, 1.0))
def test_rand_uniform_high(high: float) -> None:
    values = rand_uniform((100000,), high=high, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() < high


def test_rand_uniform_mock() -> None:
    with patch(
        "startorch.random.continuous_bounded.torch.rand",
        Mock(return_value=torch.tensor([0.0, 0.5, 1.0])),
    ):
        assert rand_uniform(size=(3,), low=1, high=5, generator=get_torch_generator(1)).equal(
            torch.tensor([1.0, 3.0, 5.0])
        )


def test_rand_uniform_incorrect_low_high() -> None:
    with raises(ValueError):
        rand_uniform((1000,), low=1, high=0.5)


def test_rand_uniform_same_random_seed() -> None:
    assert rand_uniform((1000,), generator=get_torch_generator(1)).equal(
        rand_uniform((1000,), generator=get_torch_generator(1))
    )


def test_rand_uniform_different_random_seeds() -> None:
    assert not rand_uniform((1000,), generator=get_torch_generator(1)).equal(
        rand_uniform((1000,), generator=get_torch_generator(2))
    )


#############################
#     Tests for uniform     #
#############################


def test_uniform_1d() -> None:
    values = uniform(torch.zeros(100000), torch.ones(100000), generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() < 1.0


def test_uniform_2d() -> None:
    values = uniform(
        torch.zeros(1000, 100), torch.ones(1000, 100), generator=get_torch_generator(1)
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() < 1.0


@mark.parametrize("low", (-1.0, 0.0, 1.0))
def test_uniform_low(low: float) -> None:
    values = uniform(
        torch.full((100000,), low), torch.full((100000,), 2.0), generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= low
    assert values.max() < 2.0


@mark.parametrize("high", (0.1, 0.5, 1.0))
def test_uniform_high(high: float) -> None:
    values = uniform(
        torch.zeros(100000), torch.full((100000,), high), generator=get_torch_generator(1)
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.0
    assert values.max() < high


def test_uniform_mock() -> None:
    with patch(
        "startorch.random.continuous_bounded.torch.rand",
        Mock(return_value=torch.tensor([0.0, 0.5, 1.0])),
    ):
        assert uniform(
            torch.ones(3), torch.full((3,), 5.0), generator=get_torch_generator(1)
        ).equal(torch.tensor([1.0, 3.0, 5.0]))


def test_uniform_incorrect_low_high() -> None:
    with raises(ValueError):
        uniform(torch.full((1000,), 1.0), torch.full((1000,), 0.5))


def test_uniform_shape_mismatch() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        uniform(torch.zeros(5), torch.ones(6), generator=get_torch_generator(1))


def test_uniform_same_random_seed() -> None:
    assert uniform(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(1)).equal(
        uniform(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(1))
    )


def test_uniform_different_random_seeds() -> None:
    assert not uniform(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(1)).equal(
        uniform(torch.zeros(1000), torch.ones(1000), generator=get_torch_generator(2))
    )


######################################
#     Tests for rand_log_uniform     #
######################################


def test_rand_log_uniform_1d() -> None:
    values = rand_log_uniform((100000,), low=0.1, high=1000.0, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(108.56276311376536), rtol=TOLERANCE)
    assert values.min() >= 0.1
    assert values.max() < 1000.0


def test_rand_log_uniform_2d() -> None:
    values = rand_log_uniform((1000, 100), low=0.1, high=1000.0, generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(108.56276311376536), rtol=TOLERANCE)
    assert values.min() >= 0.1
    assert values.max() < 1000.0


@mark.parametrize("low", (0.01, 0.1, 1.0))
def test_rand_log_uniform_low(low: float) -> None:
    values = rand_log_uniform((100000,), low=low, high=10.0, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(
        torch.tensor(10.0 - low).div(math.log(10.0 / low)), rtol=TOLERANCE
    )
    assert values.min() >= low
    assert values.max() < 10.0


@mark.parametrize("high", (1.0, 10.0, 100.0))
def test_rand_log_uniform_high(high: float) -> None:
    values = rand_log_uniform((100000,), low=0.1, high=high, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(
        torch.tensor(high - 0.1).div(math.log(high / 0.1)), rtol=TOLERANCE
    )
    assert values.min() >= 0.1
    assert values.max() < high


def test_rand_log_uniform_mock() -> None:
    with patch(
        "startorch.random.continuous_bounded.torch.rand",
        Mock(return_value=torch.tensor([0.0, 0.5, 1.0])),
    ):
        assert rand_log_uniform(
            size=(3,), low=1.0, high=100.0, generator=get_torch_generator(1)
        ).allclose(torch.tensor([1.0, 10.0, 100.0], dtype=torch.float), atol=TOLERANCE)


def test_rand_log_uniform_incorrect_low_high() -> None:
    with raises(ValueError, match="`high` (.*) has to be greater or equal to `low` (.*)"):
        rand_log_uniform((1000,), low=1, high=0.5)


def test_rand_log_uniform_same_random_seed() -> None:
    assert rand_log_uniform((1000,), low=0.1, high=1000.0, generator=get_torch_generator(1)).equal(
        rand_log_uniform((1000,), low=0.1, high=1000.0, generator=get_torch_generator(1))
    )


def test_rand_log_uniform_different_random_seeds() -> None:
    assert not rand_log_uniform(
        (1000,), low=0.1, high=1000.0, generator=get_torch_generator(1)
    ).equal(rand_log_uniform((1000,), low=0.1, high=1000.0, generator=get_torch_generator(2)))


#################################
#     Tests for log_uniform     #
#################################


def test_log_uniform_1d() -> None:
    values = log_uniform(
        torch.full((100000,), 0.1),
        torch.full((100000,), 1000.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(108.56276311376536), rtol=TOLERANCE)
    assert values.min() >= 0.1
    assert values.max() < 1000.0


def test_log_uniform_2d() -> None:
    values = log_uniform(
        torch.full((1000, 100), 0.1),
        torch.full((1000, 100), 1000.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(108.56276311376536), rtol=TOLERANCE)
    assert values.min() >= 0.1
    assert values.max() < 1000.0


@mark.parametrize("low", (0.01, 0.1, 1.0))
def test_log_uniform_low(low: float) -> None:
    values = log_uniform(
        torch.full((100000,), low),
        torch.full((100000,), 10.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(
        torch.tensor(10.0 - low).div(math.log(10.0 / low)), rtol=TOLERANCE
    )
    assert values.min() >= low
    assert values.max() < 10.0


@mark.parametrize("high", (1.0, 10.0, 100.0))
def test_log_uniform_high(high: float) -> None:
    values = log_uniform(
        torch.full((100000,), 0.1),
        torch.full((100000,), high),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(
        torch.tensor(high - 0.1).div(math.log(high / 0.1)), rtol=TOLERANCE
    )
    assert values.min() >= 0.1
    assert values.max() < high


def test_log_uniform_mock() -> None:
    with patch(
        "startorch.random.continuous_bounded.torch.rand",
        Mock(return_value=torch.tensor([0.0, 0.5, 1.0])),
    ):
        assert log_uniform(
            torch.ones(3), torch.full((3,), 100.0), generator=get_torch_generator(1)
        ).allclose(torch.tensor([1.0, 10.0, 100.0], dtype=torch.float), atol=TOLERANCE)


def test_log_uniform_incorrect_low_high() -> None:
    with raises(
        ValueError,
        match=(
            "Found at least one value in `low` that is higher than its associated "
            "value in `high`"
        ),
    ):
        log_uniform(torch.full((1000,), 2.0), torch.full((1000,), 0.5))


def test_log_uniform_shape_mismatch() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        log_uniform(torch.full((5,), 0.01), torch.ones(6), generator=get_torch_generator(1))


def test_log_uniform_same_random_seed() -> None:
    assert log_uniform(
        torch.ones(1000),
        torch.full((1000,), 100.0),
        generator=get_torch_generator(1),
    ).equal(
        log_uniform(
            torch.ones(1000),
            torch.full((1000,), 100.0),
            generator=get_torch_generator(1),
        )
    )


def test_log_uniform_different_random_seeds() -> None:
    assert not log_uniform(
        torch.ones(1000),
        torch.full((1000,), 100.0),
        generator=get_torch_generator(1),
    ).equal(
        log_uniform(
            torch.ones(1000),
            torch.full((1000,), 100.0),
            generator=get_torch_generator(2),
        )
    )


########################################
#     Tests for rand_asinh_uniform     #
########################################


def test_rand_asinh_uniform_1d() -> None:
    values = rand_asinh_uniform((100000,), low=-10.0, high=10.0, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.min() >= -10.0
    assert values.max() < 10.0


def test_rand_asinh_uniform_2d() -> None:
    values = rand_asinh_uniform((1000, 100), low=-10.0, high=10.0, generator=get_torch_generator(1))
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.min() >= -10.0
    assert values.max() < 10.0


@mark.parametrize("low", (-1.0, 0.0, 1.0))
def test_rand_asinh_uniform_low(low: float) -> None:
    values = rand_asinh_uniform((100000,), low=low, high=10.0, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= low
    assert values.max() < 10.0


@mark.parametrize("high", (1.0, 10.0, 100.0))
def test_rand_asinh_uniform_high(high: float) -> None:
    values = rand_asinh_uniform((100000,), low=0.1, high=high, generator=get_torch_generator(1))
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.1
    assert values.max() < high


def test_rand_asinh_uniform_mock() -> None:
    with patch(
        "startorch.random.continuous_bounded.torch.rand",
        Mock(return_value=torch.tensor([0.0, 0.5, 1.0])),
    ):
        assert rand_asinh_uniform(
            size=(3,), low=-100.0, high=100.0, generator=get_torch_generator(1)
        ).allclose(torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float), atol=TOLERANCE)


def test_rand_asinh_uniform_incorrect_low_high() -> None:
    with raises(ValueError, match="`high` (.*) has to be greater or equal to `low` (.*)"):
        rand_asinh_uniform((1000,), low=1.0, high=0.5)


def test_rand_asinh_uniform_same_random_seed() -> None:
    assert rand_asinh_uniform(
        (1000,), low=0.1, high=1000.0, generator=get_torch_generator(1)
    ).equal(rand_asinh_uniform((1000,), low=0.1, high=1000.0, generator=get_torch_generator(1)))


def test_rand_asinh_uniform_different_random_seeds() -> None:
    assert not rand_asinh_uniform(
        (1000,), low=0.1, high=1000.0, generator=get_torch_generator(1)
    ).equal(rand_asinh_uniform((1000,), low=0.1, high=1000.0, generator=get_torch_generator(2)))


###################################
#     Tests for asinh_uniform     #
###################################


def test_asinh_uniform_1d() -> None:
    values = asinh_uniform(
        torch.full((100000,), -10.0),
        torch.full((100000,), 10.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.min() >= -10.0
    assert values.max() < 10.0


def test_asinh_uniform_2d() -> None:
    values = asinh_uniform(
        torch.full((1000, 100), -10.0),
        torch.full((1000, 100), 10.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (1000, 100)
    assert values.dtype == torch.float
    assert values.mean().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.median().allclose(torch.tensor(0.0), atol=TOLERANCE)
    assert values.min() >= -10.0
    assert values.max() < 10.0


@mark.parametrize("low", (-1.0, 0.0, 1.0))
def test_asinh_uniform_low(low: float) -> None:
    values = asinh_uniform(
        torch.full((100000,), low),
        torch.full((100000,), 10.0),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= low
    assert values.max() < 10.0


@mark.parametrize("high", (1.0, 10.0, 100.0))
def test_asinh_uniform_high(high: float) -> None:
    values = asinh_uniform(
        torch.full((100000,), 0.1),
        torch.full((100000,), high),
        generator=get_torch_generator(1),
    )
    assert values.shape == (100000,)
    assert values.dtype == torch.float
    assert values.min() >= 0.1
    assert values.max() < high


def test_asinh_uniform_mock() -> None:
    with patch(
        "startorch.random.continuous_bounded.torch.rand",
        Mock(return_value=torch.tensor([0.0, 0.5, 1.0])),
    ):
        assert asinh_uniform(
            torch.full((3,), -100.0), torch.full((3,), 100.0), generator=get_torch_generator(1)
        ).allclose(torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float), atol=TOLERANCE)


def test_asinh_uniform_incorrect_low_high() -> None:
    with raises(
        ValueError,
        match=(
            "Found at least one value in `low` that is higher than its associated "
            "value in `high`"
        ),
    ):
        asinh_uniform(torch.full((1000,), 2.0), torch.full((1000,), 0.5))


def test_asinh_uniform_shape_mismatch() -> None:
    with raises(
        ValueError, match="Incorrect shapes. The shapes of all the input tensors must be equal:"
    ):
        asinh_uniform(torch.zeros(5), torch.ones(6), generator=get_torch_generator(1))


def test_asinh_uniform_same_random_seed() -> None:
    assert asinh_uniform(
        torch.full((1000,), -100.0),
        torch.full((1000,), 100.0),
        generator=get_torch_generator(1),
    ).equal(
        asinh_uniform(
            torch.full((1000,), -100.0),
            torch.full((1000,), 100.0),
            generator=get_torch_generator(1),
        )
    )


def test_asinh_uniform_different_random_seeds() -> None:
    assert not asinh_uniform(
        torch.full((1000,), -100.0),
        torch.full((1000,), 100.0),
        generator=get_torch_generator(1),
    ).equal(
        asinh_uniform(
            torch.full((1000,), -100.0),
            torch.full((1000,), 100.0),
            generator=get_torch_generator(2),
        )
    )
