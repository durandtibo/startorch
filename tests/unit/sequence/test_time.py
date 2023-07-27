from pytest import mark, raises
from redcat import BatchedTensorSeq

from startorch.sequence import Time

SIZES = (1, 2)


##########################
#     Tests for Time     #
##########################


def test_time_str() -> None:
    assert str(Time.create_uniform_time()).startswith("TimeSequenceGenerator(")


@mark.parametrize(
    "generator",
    (
        Time.create_exponential_constant_time_diff(),
        Time.create_exponential_time_diff(),
        Time.create_poisson_constant_time_diff(),
        Time.create_poisson_time_diff(),
        Time.create_uniform_constant_time_diff(),
        Time.create_uniform_time_diff(),
        Time.create_uniform_time(),
    ),
)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_time_generate(generator: Time, batch_size: int, seq_len: int) -> None:
    batch = generator.generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len


@mark.parametrize("min_time_diff", (-0.1, -1))
def test_time_generate_uniform_time_diff_incorrect_min_time_diff(min_time_diff: float) -> None:
    with raises(ValueError, match="min_time_diff has to be greater or equal to 0"):
        Time.create_uniform_time_diff(min_time_diff=min_time_diff, max_time_diff=1)


@mark.parametrize("min_time_diff", (-0.1, -1))
def test_time_generate_uniform_constant_time_diff_incorrect_min_time_diff(
    min_time_diff: float,
) -> None:
    with raises(ValueError, match="min_time_diff has to be greater or equal to 0"):
        Time.create_uniform_constant_time_diff(min_time_diff=min_time_diff, max_time_diff=1)


@mark.parametrize("min_time", (-0.1, -1))
def test_time_generate_uniform_time_incorrect_min_time(min_time: float) -> None:
    with raises(ValueError, match="min_time has to be greater or equal to 0"):
        Time.create_uniform_time(min_time=min_time)
