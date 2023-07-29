from __future__ import annotations

from collections.abc import Sequence

import torch
from pytest import mark, raises
from torch import Tensor

from startorch.tensor import Multinomial, UniformCategorical
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


#################################
#     Tests for Multinomial     #
#################################


def test_multinomial_str() -> None:
    assert str(Multinomial(torch.ones(10))).startswith("MultinomialTensorGenerator(")


@mark.parametrize(
    "weights", [torch.ones(10), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
)
@mark.parametrize("size", SIZES)
def test_multinomial_generate(weights: Tensor | Sequence, size: tuple[int, ...]) -> None:
    tensor = Multinomial(weights).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.long
    assert tensor.min() >= 0
    assert tensor.max() < 10


@mark.parametrize("num_categories", (1, 2, 4))
def test_multinomial_generate_num_categories(num_categories: int) -> None:
    tensor = Multinomial(torch.ones(num_categories)).generate(size=(4, 10))
    assert tensor.shape == (4, 10)
    assert tensor.dtype == torch.long
    assert tensor.min() >= 0
    assert tensor.max() < num_categories


def test_multinomial_generate_same_random_seed() -> None:
    generator = Multinomial(torch.ones(10))
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_multinomial_generate_different_random_seeds() -> None:
    generator = Multinomial(torch.ones(10))
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )


@mark.parametrize(
    "generator",
    (
        Multinomial.create_uniform_weights(num_categories=10),
        Multinomial.create_exp_weights(num_categories=10),
        Multinomial.create_linear_weights(num_categories=10),
    ),
)
@mark.parametrize("size", SIZES)
def test_multinomial_generate_predefined_settings(
    generator: Multinomial, size: tuple[int, ...]
) -> None:
    tensor = generator.generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.long
    assert tensor.min() >= 0
    assert tensor.max() < 10


########################################
#     Tests for UniformCategorical     #
########################################


def test_uniform_categorical_str() -> None:
    assert str(UniformCategorical(num_categories=50)).startswith(
        "UniformCategoricalTensorGenerator("
    )


@mark.parametrize("num_categories", (1, 2))
def test_uniform_categorical_num_categories(num_categories: int) -> None:
    assert UniformCategorical(num_categories)._num_categories == num_categories


@mark.parametrize("num_categories", (0, -1))
def test_uniform_categorical_incorrect_num_categories(num_categories: int) -> None:
    with raises(ValueError, match="num_categories has to be greater than 0"):
        UniformCategorical(num_categories)


@mark.parametrize("size", SIZES)
def test_uniform_categorical_generate(size: tuple[int, ...]) -> None:
    tensor = UniformCategorical(num_categories=50).generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.long
    assert tensor.min() >= 0
    assert tensor.max() < 50


def test_uniform_categorical_generate_same_random_seed() -> None:
    generator = UniformCategorical(num_categories=50)
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_uniform_categorical_generate_different_random_seeds() -> None:
    generator = UniformCategorical(num_categories=50)
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
