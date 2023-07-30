from __future__ import annotations

import torch
from pytest import mark

from startorch.tensor import MultinomialChoice, RandNormal, RandUniform
from startorch.utils.seed import get_torch_generator
from startorch.utils.weight import GENERATOR, WEIGHT

SIZES = ((1,), (2, 3), (2, 3, 4))


#######################################
#     Tests for MultinomialChoice     #
#######################################


def test_multinomial_choice_str() -> None:
    assert str(
        MultinomialChoice(
            (
                {WEIGHT: 2.0, GENERATOR: RandUniform()},
                {WEIGHT: 1.0, GENERATOR: RandNormal()},
            )
        )
    ).startswith("MultinomialChoiceTensorGenerator(")


def test_multinomial_choice_generators() -> None:
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
        )
    )
    assert len(generator._tensors) == 2
    assert isinstance(generator._tensors[0], RandUniform)
    assert isinstance(generator._tensors[1], RandNormal)


def test_multinomial_choice_weights() -> None:
    assert MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
            {WEIGHT: 3.0, GENERATOR: RandUniform()},
        )
    )._weights.equal(torch.tensor([2.0, 1.0, 3.0]))


@mark.parametrize("size", SIZES)
def test_multinomial_choice_generate(size: tuple[int, ...]) -> None:
    tensor = MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
        ),
    ).generate(size)
    assert tensor.shape == size


def test_multinomial_choice_generate_same_random_seed() -> None:
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
        ),
    )
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_multinomial_choice_generate_different_random_seeds() -> None:
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
        ),
    )
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
