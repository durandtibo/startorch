from __future__ import annotations

import torch
from eternity.constants import CREATOR, WEIGHT
from eternity.generators.tensor import BaseTensorGenerator, MultinomialChoice
from pytest import mark

from tests.unit.generators.configs import generate_tensor_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


#######################################
#     Tests for MultinomialChoice     #
#######################################


def test_multinomial_choice_str():
    assert str(
        MultinomialChoice(
            (
                {WEIGHT: 2.0, CREATOR: generate_tensor_generator()},
                {WEIGHT: 1.0, CREATOR: generate_tensor_generator()},
            )
        )
    ).startswith("MultinomialChoiceTensorGenerator(")


def test_multinomial_choice_generators():
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, CREATOR: generate_tensor_generator()},
            {WEIGHT: 1.0, CREATOR: generate_tensor_generator()},
        )
    )
    assert len(generator._generators) == 2
    assert isinstance(generator._generators[0], BaseTensorGenerator)
    assert isinstance(generator._generators[1], BaseTensorGenerator)


def test_multinomial_choice_weights():
    assert MultinomialChoice(
        (
            {WEIGHT: 2.0, CREATOR: generate_tensor_generator()},
            {WEIGHT: 1.0, CREATOR: generate_tensor_generator()},
            {WEIGHT: 3.0, CREATOR: generate_tensor_generator()},
        )
    )._weights.equal(torch.tensor([2.0, 1.0, 3.0]))


@mark.parametrize("random_seed", (1, 2))
def test_multinomial_choice_random_seed_auto_set_random_seed_true(
    random_seed: int,
):
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, CREATOR: generate_tensor_generator(random_seed=101)},
            {WEIGHT: 1.0, CREATOR: generate_tensor_generator(random_seed=102)},
            {WEIGHT: 3.0, CREATOR: generate_tensor_generator(random_seed=103)},
        ),
        random_seed=random_seed,
    )
    assert generator.get_random_seed() == random_seed
    assert (
        len(
            {
                generator.get_random_seed(),
                generator._generators[0].get_random_seed(),
                generator._generators[1].get_random_seed(),
                generator._generators[2].get_random_seed(),
            }
        )
        == 4
    )


@mark.parametrize("random_seed", (1, 2))
def test_multinomial_choice_random_seed_auto_set_random_seed_false(
    random_seed: int,
):
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, CREATOR: generate_tensor_generator(random_seed=101)},
            {WEIGHT: 1.0, CREATOR: generate_tensor_generator(random_seed=102)},
            {WEIGHT: 3.0, CREATOR: generate_tensor_generator(random_seed=103)},
        ),
        random_seed=random_seed,
        auto_set_random_seed=False,
    )
    assert generator.get_random_seed() == random_seed
    assert generator._generators[0].get_random_seed() == 101
    assert generator._generators[1].get_random_seed() == 102
    assert generator._generators[2].get_random_seed() == 103


@mark.parametrize("random_seed", (1, 2))
def test_multinomial_choice_set_random_seed_auto_set_random_seed_true(
    random_seed: int,
):
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, CREATOR: generate_tensor_generator(random_seed=101)},
            {WEIGHT: 1.0, CREATOR: generate_tensor_generator(random_seed=102)},
            {WEIGHT: 3.0, CREATOR: generate_tensor_generator(random_seed=103)},
        ),
    )
    generator.set_random_seed(random_seed)
    assert generator.get_random_seed() == random_seed
    assert (
        len(
            {
                generator.get_random_seed(),
                generator._generators[0].get_random_seed(),
                generator._generators[1].get_random_seed(),
                generator._generators[2].get_random_seed(),
            }
        )
        == 4
    )


@mark.parametrize("random_seed", (1, 2))
def test_multinomial_choice_set_random_seed_auto_set_random_seed_false(
    random_seed: int,
):
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, CREATOR: generate_tensor_generator(random_seed=101)},
            {WEIGHT: 1.0, CREATOR: generate_tensor_generator(random_seed=102)},
            {WEIGHT: 3.0, CREATOR: generate_tensor_generator(random_seed=103)},
        ),
        auto_set_random_seed=False,
    )
    generator.set_random_seed(random_seed)
    assert generator.get_random_seed() == random_seed
    assert generator._generators[0].get_random_seed() == 101
    assert generator._generators[1].get_random_seed() == 102
    assert generator._generators[2].get_random_seed() == 103


@mark.parametrize("size", SIZES)
def test_multinomial_choice_generate(size: tuple[int, ...]):
    tensor = MultinomialChoice(
        (
            {WEIGHT: 2.0, CREATOR: generate_tensor_generator(random_seed=101)},
            {WEIGHT: 1.0, CREATOR: generate_tensor_generator(random_seed=102)},
        ),
    ).generate(size)
    assert tensor.shape == size


def test_multinomial_choice_generate_same_random_seed():
    assert (
        MultinomialChoice(
            (
                {WEIGHT: 2.0, CREATOR: generate_tensor_generator()},
                {WEIGHT: 1.0, CREATOR: generate_tensor_generator()},
            ),
            random_seed=1,
        )
        .generate(size=(4, 12))
        .equal(
            MultinomialChoice(
                (
                    {WEIGHT: 2.0, CREATOR: generate_tensor_generator()},
                    {WEIGHT: 1.0, CREATOR: generate_tensor_generator()},
                ),
                random_seed=1,
            ).generate(size=(4, 12))
        )
    )


def test_multinomial_choice_generate_different_random_seeds():
    assert (
        not MultinomialChoice(
            (
                {WEIGHT: 2.0, CREATOR: generate_tensor_generator()},
                {WEIGHT: 1.0, CREATOR: generate_tensor_generator()},
            ),
            random_seed=1,
        )
        .generate(size=(4, 12))
        .equal(
            MultinomialChoice(
                (
                    {WEIGHT: 2.0, CREATOR: generate_tensor_generator()},
                    {WEIGHT: 1.0, CREATOR: generate_tensor_generator()},
                ),
                random_seed=2,
            ).generate(size=(4, 12))
        )
    )
