import torch
from pytest import mark, raises
from redcat import BatchedTensorSeq

from startorch.sequence import RandInt, RandUniform
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2)


#############################
#     Tests for RandInt     #
#############################


def test_rand_int_str() -> None:
    assert str(RandInt(low=5, high=20)).startswith("RandIntSequenceGenerator(")


@mark.parametrize("low", (1, 2))
def test_rand_int_low(low: int) -> None:
    assert RandInt(low=low, high=20)._low == low


@mark.parametrize("high", (10, 20))
def test_rand_int_high(high: int) -> None:
    assert RandInt(low=0, high=high)._high == high


def test_rand_int_incorrect_min_max() -> None:
    with raises(ValueError):
        RandInt(low=5, high=4)


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_rand_int_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandInt(low=5, high=20).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len)
    assert batch.data.dtype == torch.long


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_rand_int_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandInt(low=5, high=20, feature_size=feature_size).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.long


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_rand_int_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandInt(low=5, high=20, feature_size=(3, 4)).generate(
        batch_size=batch_size, seq_len=seq_len
    )
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.long


@mark.parametrize("low", (1, 5))
@mark.parametrize("high", (10, 20))
def test_rand_int_generate_high(low: int, high: int) -> None:
    batch = RandInt(low=low, high=high).generate(batch_size=10, seq_len=10)
    assert batch.min() >= low
    assert batch.max() < high


def test_rand_int_generate_high_1() -> None:
    assert (
        RandInt(low=1, high=2)
        .generate(batch_size=2, seq_len=3)
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long)))
    )


def test_rand_int_generate_same_random_seed() -> None:
    generator = RandInt(low=5, high=20)
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_int_generate_different_random_seeds() -> None:
    generator = RandInt(low=5, high=20)
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )


#################################
#     Tests for RandUniform     #
#################################


def test_rand_uniform_str() -> None:
    assert str(RandUniform()).startswith("RandUniformSequenceGenerator(")


@mark.parametrize("low", (1.0, 2.0))
def test_rand_uniform_low(low: float) -> None:
    assert RandUniform(low=low, high=10)._low == low


@mark.parametrize("high", (1.0, 10.0))
def test_rand_uniform_high(high: float) -> None:
    assert RandUniform(low=1, high=high)._high == high


def test_rand_uniform_incorrect_min_high() -> None:
    with raises(ValueError, match="high (.*) has to be greater or equal to low (.*)"):
        RandUniform(low=2, high=1)


def test_rand_uniform_feature_size_default() -> None:
    assert RandUniform()._feature_size == (1,)


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_rand_uniform_generate_feature_size_default(batch_size: int, seq_len: int) -> None:
    batch = RandUniform().generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 1)
    assert batch.data.dtype == torch.float


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_rand_uniform_generate_feature_size_int(
    batch_size: int, seq_len: int, feature_size: int
) -> None:
    batch = RandUniform(feature_size=feature_size).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, feature_size)
    assert batch.data.dtype == torch.float


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_rand_uniform_generate_feature_size_tuple(batch_size: int, seq_len: int) -> None:
    batch = RandUniform(feature_size=(3, 4)).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.batch_size == batch_size
    assert batch.seq_len == seq_len
    assert batch.data.shape == (batch_size, seq_len, 3, 4)
    assert batch.data.dtype == torch.float


def test_rand_uniform_value_1() -> None:
    assert (
        RandUniform(low=1, high=1)
        .generate(batch_size=2, seq_len=4)
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [[[1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0]]], dtype=torch.float
                )
            )
        )
    )


def test_rand_uniform_generate_same_random_seed() -> None:
    generator = RandUniform()
    assert generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1))
    )


def test_rand_uniform_generate_different_random_seeds() -> None:
    generator = RandUniform()
    assert not generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2))
    )