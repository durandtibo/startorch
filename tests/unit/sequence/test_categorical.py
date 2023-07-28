from __future__ import annotations

import torch
from pytest import mark, raises
from redcat import BatchedTensorSeq

from startorch.sequence import Multinomial, UniformCategorical
from startorch.sequence.categorical import prepare_probabilities

SIZES = (1, 2)


#################################
#     Tests for Multinomial     #
#################################


def test_multinomial_str():
    assert str(Multinomial(torch.ones(10))).startswith("MultinomialSequenceGenerator(")


@mark.parametrize("random_seed", (1, 2))
def test_multinomial_get_random_seed(random_seed: int):
    assert Multinomial(torch.ones(10), random_seed=random_seed).get_random_seed() == random_seed


@mark.parametrize("random_seed", (1, 2))
def test_multinomial_set_random_seed(random_seed: int):
    generator = Multinomial(torch.ones(10))
    generator.set_random_seed(random_seed)
    assert generator.get_random_seed() == random_seed


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_multinomial_generate_feature_size_default(batch_size: int, seq_len: int):
    batch = Multinomial(torch.ones(10)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 10


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_multinomial_generate_feature_size_int(batch_size: int, seq_len: int, feature_size: int):
    batch = Multinomial(torch.ones(10), feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 10


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_multinomial_generate_feature_size_tuple(batch_size: int, seq_len: int):
    batch = Multinomial(torch.ones(10), feature_size=(3, 4)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 10


@mark.parametrize("num_categories", SIZES)
def test_multinomial_generate_num_categories(num_categories: int):
    batch = Multinomial(torch.ones(num_categories)).generate(batch_size=4, seq_len=10)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == 4
    assert batch.seq_len == 10
    assert batch.data.shape == (4, 10, 1)
    assert batch.data.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < num_categories


def test_multinomial_generate_same_random_seed():
    assert (
        Multinomial(torch.ones(10), random_seed=1)
        .generate(batch_size=4, seq_len=12)
        .equal(Multinomial(torch.ones(10), random_seed=1).generate(batch_size=4, seq_len=12))
    )


def test_multinomial_generate_different_random_seeds():
    assert (
        not Multinomial(torch.ones(10), random_seed=1)
        .generate(batch_size=4, seq_len=12)
        .equal(Multinomial(torch.ones(10), random_seed=2).generate(batch_size=4, seq_len=12))
    )


@mark.parametrize(
    "generator",
    (
        Multinomial.generate_uniform_weights(num_categories=10),
        Multinomial.generate_exp_weights(num_categories=10),
        Multinomial.generate_linear_weights(num_categories=10),
    ),
)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_multinomial_generate_predefined_settings(
    generator: Multinomial, batch_size: int, seq_len: int
):
    batch = generator.generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 10


###########################################
#     Tests for prepare_probabilities     #
###########################################


@mark.parametrize(
    "weights,probabilities",
    (
        (torch.ones(1), torch.ones(1)),
        (torch.ones(2), 0.5 * torch.ones(2)),
        (torch.ones(4), 0.25 * torch.ones(4)),
        (torch.ones(8), 0.125 * torch.ones(8)),
        (0.1 * torch.ones(4), 0.25 * torch.ones(4)),
        (
            torch.tensor([1, 2, 4, 7], dtype=torch.float),
            torch.tensor(
                [0.07142857142857142, 0.14285714285714285, 0.2857142857142857, 0.5],
                dtype=torch.float,
            ),
        ),
    ),
)
def test_prepare_probabilities(weights: torch.Tensor, probabilities: torch.Tensor):
    assert torch.allclose(prepare_probabilities(weights), probabilities)


@mark.parametrize(
    "weights",
    (
        torch.tensor([-1, 2, 4, 7], dtype=torch.float),
        torch.zeros(5),
        torch.ones(4, 5),
    ),
)
def test_prepare_probabilities_incorrect_weights(weights):
    with raises(ValueError):
        prepare_probabilities(weights)


########################################
#     Tests for UniformCategorical     #
########################################


def test_uniform_categorical_str():
    assert str(UniformCategorical(num_categories=50)).startswith(
        "UniformCategoricalSequenceGenerator("
    )


@mark.parametrize("num_categories", (1, 2))
def test_uniform_categorical_num_categories(num_categories: int):
    assert UniformCategorical(num_categories)._num_categories == num_categories


@mark.parametrize("num_categories", (0, -1))
def test_uniform_categorical_incorrect_num_categories(num_categories: int):
    with raises(ValueError):
        UniformCategorical(num_categories)


def test_uniform_categorical_feature_size_default():
    assert UniformCategorical(num_categories=50)._feature_size == tuple()


@mark.parametrize("random_seed", (1, 2))
def test_uniform_categorical_get_random_seed(random_seed: int):
    assert (
        UniformCategorical(num_categories=50, random_seed=random_seed).get_random_seed()
        == random_seed
    )


@mark.parametrize("random_seed", (1, 2))
def test_uniform_categorical_set_random_seed(random_seed: int):
    generator = UniformCategorical(num_categories=50)
    generator.set_random_seed(random_seed)
    assert generator.get_random_seed() == random_seed


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_uniform_categorical_generate_feature_size_default(batch_size: int, seq_len: int):
    batch = UniformCategorical(num_categories=50).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len)
    assert batch.data.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 50


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_uniform_categorical_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
):
    batch = UniformCategorical(num_categories=50, feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 50


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_uniform_categorical_generate_feature_size_tuple(batch_size: int, seq_len: int):
    batch = UniformCategorical(num_categories=50, feature_size=(3, 4)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.long
    assert batch.min() >= 0
    assert batch.max() < 50


def test_uniform_categorical_generate_same_random_seed():
    assert (
        UniformCategorical(num_categories=50, random_seed=1)
        .generate(batch_size=4, seq_len=12)
        .equal(
            UniformCategorical(num_categories=50, random_seed=1).generate(batch_size=4, seq_len=12)
        )
    )


def test_uniform_categorical_generate_different_random_seeds():
    assert (
        not UniformCategorical(num_categories=50, random_seed=1)
        .generate(batch_size=4, seq_len=12)
        .equal(
            UniformCategorical(num_categories=50, random_seed=2).generate(batch_size=4, seq_len=12)
        )
    )
