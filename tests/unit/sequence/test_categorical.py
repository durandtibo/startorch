from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

from startorch.sequence import Multinomial, UniformCategorical
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]

if TYPE_CHECKING:
    from collections.abc import Sequence


#################################
#     Tests for Multinomial     #
#################################


def test_multinomial_str() -> None:
    assert str(Multinomial(torch.ones(10))).startswith("MultinomialSequenceGenerator(")


@pytest.mark.parametrize(
    "weights", [torch.ones(10), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_multinomial_generate_feature_size_default(
    weights: Tensor | Sequence, batch_size: int, seq_len: int
) -> None:
    batch = Multinomial(weights).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 1)
    assert batch.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 10


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_multinomial_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = Multinomial(torch.ones(10), feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 10


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_multinomial_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = Multinomial(torch.ones(10), feature_size=(3, 4)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 3, 4)
    assert batch.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 10


@pytest.mark.parametrize("num_categories", SIZES)
def test_multinomial_generate_num_categories(num_categories: int) -> None:
    batch = Multinomial(torch.ones(num_categories)).generate(batch_size=4, seq_len=10)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 10, 1)
    assert batch.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < num_categories


def test_multinomial_generate_same_random_seed() -> None:
    generator = Multinomial(torch.ones(10))
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_multinomial_generate_different_random_seeds() -> None:
    generator = Multinomial(torch.ones(10))
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


@pytest.mark.parametrize(
    "generator",
    [
        Multinomial.create_uniform_weights(num_categories=10),
        Multinomial.create_exp_weights(num_categories=10),
        Multinomial.create_linear_weights(num_categories=10),
    ],
)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_multinomial_generate_predefined_settings(
    generator: Multinomial, batch_size: int, seq_len: int
) -> None:
    batch = generator.generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 1)
    assert batch.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 10


########################################
#     Tests for UniformCategorical     #
########################################


def test_uniform_categorical_str() -> None:
    assert str(UniformCategorical(num_categories=50)).startswith(
        "UniformCategoricalSequenceGenerator("
    )


@pytest.mark.parametrize("num_categories", [1, 2])
def test_uniform_categorical_num_categories(num_categories: int) -> None:
    assert UniformCategorical(num_categories)._num_categories == num_categories


@pytest.mark.parametrize("num_categories", [0, -1])
def test_uniform_categorical_incorrect_num_categories(num_categories: int) -> None:
    with pytest.raises(ValueError, match=r"num_categories has to be greater than 0"):
        UniformCategorical(num_categories)


def test_uniform_categorical_feature_size_default() -> None:
    assert UniformCategorical(num_categories=50)._feature_size == ()


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_uniform_categorical_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = UniformCategorical(num_categories=50).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len)
    assert batch.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 50


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_uniform_categorical_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = UniformCategorical(num_categories=50, feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 50


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_uniform_categorical_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = UniformCategorical(num_categories=50, feature_size=(3, 4)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 3, 4)
    assert batch.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 50


def test_uniform_categorical_generate_same_random_seed() -> None:
    generator = UniformCategorical(num_categories=50)
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_uniform_categorical_generate_different_random_seeds() -> None:
    generator = UniformCategorical(num_categories=50)
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
