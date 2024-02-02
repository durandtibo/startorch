from __future__ import annotations

import pytest
import torch

from startorch.sequence import Time

SIZES = [1, 2, 4]


##########################
#     Tests for Time     #
##########################


def test_time_str() -> None:
    assert str(Time.create_uniform_time()).startswith("TimeSequenceGenerator(")


@pytest.mark.parametrize(
    "generator",
    [
        Time.create_exponential_constant_time_diff(),
        Time.create_exponential_time_diff(),
        Time.create_poisson_constant_time_diff(),
        Time.create_poisson_time_diff(),
        Time.create_uniform_constant_time_diff(),
        Time.create_uniform_time_diff(),
        Time.create_uniform_time(),
    ],
)
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
def test_time_generate(generator: Time, batch_size: int, seq_len: int) -> None:
    batch = generator.generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, 1)


@pytest.mark.parametrize("min_time_diff", [-0.1, -1])
def test_time_generate_uniform_time_diff_incorrect_min_time_diff(min_time_diff: float) -> None:
    with pytest.raises(ValueError, match="min_time_diff has to be greater or equal to 0"):
        Time.create_uniform_time_diff(min_time_diff=min_time_diff, max_time_diff=1)


@pytest.mark.parametrize("min_time_diff", [-0.1, -1])
def test_time_generate_uniform_constant_time_diff_incorrect_min_time_diff(
    min_time_diff: float,
) -> None:
    with pytest.raises(ValueError, match="min_time_diff has to be greater or equal to 0"):
        Time.create_uniform_constant_time_diff(min_time_diff=min_time_diff, max_time_diff=1)


@pytest.mark.parametrize("min_time", [-0.1, -1])
def test_time_generate_uniform_time_incorrect_min_time(min_time: float) -> None:
    with pytest.raises(ValueError, match="min_time has to be greater or equal to 0"):
        Time.create_uniform_time(min_time=min_time)
