from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.sequence import MarkovChain
from startorch.sequence.markov import make_markov_chain
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)

#################################
#     Tests for MarkovChain     #
#################################


def test_markov_chain_str() -> None:
    assert str(MarkovChain(transition=torch.rand(6, 6))).startswith("MarkovChainSequenceGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_markov_chain_generate_no_init(batch_size: int, seq_len: int) -> None:
    out = MarkovChain(transition=torch.rand(6, 6)).generate(batch_size=batch_size, seq_len=seq_len)
    assert out.dtype == torch.long
    assert out.shape == (batch_size, seq_len)


def test_markov_chain_generate_init_1d() -> None:
    transition = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    out = MarkovChain(transition=transition, init=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])).generate(
        batch_size=2, seq_len=5
    )
    assert objects_are_equal(out, torch.tensor([[0, 1, 3, 4, 2], [0, 1, 3, 4, 2]]))


def test_markov_chain_generate_init_2d() -> None:
    transition = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    out = MarkovChain(
        transition=transition,
        init=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]),
    ).generate(batch_size=2, seq_len=5)
    assert objects_are_equal(out, torch.tensor([[0, 1, 3, 4, 2], [4, 2, 0, 1, 3]]))


def test_markov_chain_generate_same_random_seed() -> None:
    generator = MarkovChain(transition=torch.rand(6, 6))
    assert objects_are_equal(
        generator.generate(batch_size=10, seq_len=16, rng=get_torch_generator(1)),
        generator.generate(batch_size=10, seq_len=16, rng=get_torch_generator(1)),
    )


def test_markov_chain_generate_different_random_seeds() -> None:
    generator = MarkovChain(transition=torch.rand(6, 6))
    assert not objects_are_equal(
        generator.generate(batch_size=10, seq_len=16, rng=get_torch_generator(1)),
        generator.generate(batch_size=10, seq_len=16, rng=get_torch_generator(2)),
    )


#######################################
#     Tests for make_markov_chain     #
#######################################


def test_make_markov_chain_no_init() -> None:
    transition = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    out = make_markov_chain(batch_size=2, seq_len=5, transition=transition)
    assert out.dtype == torch.long
    assert out.shape == (2, 5)


def test_make_markov_chain_init_1d() -> None:
    transition = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    out = make_markov_chain(
        batch_size=2, seq_len=5, transition=transition, init=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert objects_are_equal(out, torch.tensor([[0, 1, 3, 4, 2], [0, 1, 3, 4, 2]]))


def test_make_markov_chain_init_2d() -> None:
    transition = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    out = make_markov_chain(
        batch_size=2,
        seq_len=5,
        transition=transition,
        init=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]),
    )
    assert objects_are_equal(out, torch.tensor([[0, 1, 3, 4, 2], [4, 2, 0, 1, 3]]))


def test_make_markov_chain_same_random_seed() -> None:
    transition = torch.rand(6, 6)
    assert objects_are_equal(
        make_markov_chain(
            batch_size=10, seq_len=16, transition=transition, rng=get_torch_generator(1)
        ),
        make_markov_chain(
            batch_size=10, seq_len=16, transition=transition, rng=get_torch_generator(1)
        ),
    )


def test_make_markov_chain_different_random_seeds() -> None:
    transition = torch.rand(6, 6)
    assert not objects_are_equal(
        make_markov_chain(
            batch_size=10, seq_len=16, transition=transition, rng=get_torch_generator(1)
        ),
        make_markov_chain(
            batch_size=10, seq_len=16, transition=transition, rng=get_torch_generator(2)
        ),
    )
