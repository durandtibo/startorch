from __future__ import annotations

from unittest.mock import patch

import torch
from pytest import mark, raises
from redcat import BatchedTensorSeq

from startorch.sequence import RandWienerProcess
from startorch.sequence.wiener import wiener_process
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


#######################################
#     Tests for RandWienerProcess     #
#######################################


def test_rand_wiener_process_str() -> None:
    assert str(RandWienerProcess()).startswith("RandWienerProcessSequenceGenerator(")


@mark.parametrize("step_size", (0.1, 1.0, 10.0))
def test_rand_wiener_process_step_size(step_size: float) -> None:
    assert RandWienerProcess(step_size=step_size)._step_size == step_size


@mark.parametrize("step_size", (-0.01, -1.0))
def test_rand_wiener_process_incorrect_min_max_value(step_size: float) -> None:
    with raises(ValueError, match="step_size has to be greater than 0"):
        RandWienerProcess(step_size=step_size)


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_rand_wiener_process_generate(batch_size: int, seq_len: int) -> None:
    batch = RandWienerProcess().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float


def test_rand_wiener_process_generate_same_random_seed() -> None:
    generator = RandWienerProcess()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_wiener_process_generate_different_random_seeds() -> None:
    generator = RandWienerProcess()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


####################################
#     Tests for wiener_process     #
####################################


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_wiener_process(batch_size: int, seq_len: int) -> None:
    out = wiener_process(batch_size=batch_size, seq_len=seq_len)
    assert out.shape == (batch_size, seq_len)
    assert out.dtype == torch.float


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_wiener_process_step_size_0(batch_size: int, seq_len: int) -> None:
    assert wiener_process(step_size=0, batch_size=batch_size, seq_len=seq_len).equal(
        torch.zeros(batch_size, seq_len)
    )


@patch("startorch.sequence.wiener.torch.randn", lambda *args, **kwargs: torch.ones(2, 4))
def test_wiener_process_step_size_1() -> None:
    assert wiener_process(step_size=1, batch_size=2, seq_len=4).equal(
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.float)
    )


@patch("startorch.sequence.wiener.torch.randn", lambda *args, **kwargs: torch.ones(2, 4))
def test_wiener_process_step_size_4() -> None:
    assert wiener_process(step_size=4, batch_size=2, seq_len=4).equal(
        torch.tensor([[0, 2, 4, 6], [0, 2, 4, 6]], dtype=torch.float)
    )


def test_wiener_process_step_size_incorrect() -> None:
    with raises(ValueError, match="step_size has to be greater than 0"):
        assert wiener_process(step_size=-1, batch_size=2, seq_len=4)
