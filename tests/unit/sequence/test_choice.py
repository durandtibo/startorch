from __future__ import annotations

import pytest
import torch
from redcat import BatchedTensorSeq

from startorch.sequence import (
    BaseSequenceGenerator,
    MultinomialChoice,
    RandNormal,
    RandUniform,
)
from startorch.utils.seed import get_torch_generator
from startorch.utils.weight import GENERATOR, WEIGHT

SIZES = (1, 2, 4)


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
    ).startswith("MultinomialChoiceSequenceGenerator(")


def test_multinomial_choice_generators() -> None:
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
        )
    )
    assert len(generator._sequences) == 2
    assert isinstance(generator._sequences[0], BaseSequenceGenerator)
    assert isinstance(generator._sequences[1], BaseSequenceGenerator)


def test_multinomial_choice_weights() -> None:
    assert MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
            {WEIGHT: 3.0, GENERATOR: RandUniform()},
        )
    )._weights.equal(torch.tensor([2.0, 1.0, 3.0]))


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_multinomial_choice_generate(batch_size: int, seq_len: int) -> None:
    batch = MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
        ),
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len


def test_multinomial_choice_generate_same_random_seed() -> None:
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
        ),
    )
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_multinomial_choice_generate_different_random_seeds() -> None:
    generator = MultinomialChoice(
        (
            {WEIGHT: 2.0, GENERATOR: RandUniform()},
            {WEIGHT: 1.0, GENERATOR: RandNormal()},
        ),
    )
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )
