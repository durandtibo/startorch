from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.sequence import AutoRegressive, Full, RandNormal, RandUniform
from startorch.tensor import RandInt
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


####################################
#     Tests for AutoRegressive     #
####################################


def test_auto_regressive_str() -> None:
    assert str(
        AutoRegressive(
            value=RandNormal(),
            coefficient=RandUniform(low=-1.0, high=1.0),
            noise=Full(0.0),
            order=RandInt(low=1, high=6),
            max_abs_value=100.0,
        )
    ).startswith("AutoRegressiveSequenceGenerator(")


@pytest.mark.parametrize("max_abs_value", [1, 10])
def test_auto_regressive_max_abs_value(max_abs_value: float) -> None:
    assert (
        AutoRegressive(
            value=RandNormal(),
            coefficient=RandUniform(
                low=-1.0,
                high=1.0,
            ),
            noise=RandNormal(),
            order=RandInt(low=1, high=6),
            max_abs_value=max_abs_value,
        )._max_abs_value
        == max_abs_value
    )


@pytest.mark.parametrize("max_abs_value", [-1, 0])
def test_auto_regressive_max_abs_value_incorrect(max_abs_value: float) -> None:
    with pytest.raises(ValueError, match="`max_abs_value` has to be positive"):
        AutoRegressive(
            value=RandNormal(),
            coefficient=RandUniform(
                low=-1.0,
                high=1.0,
            ),
            noise=RandNormal(),
            order=RandInt(low=1, high=6),
            max_abs_value=max_abs_value,
        )


@pytest.mark.parametrize("warmup", [0, 1])
def test_auto_regressive_warmup(warmup: int) -> None:
    assert (
        AutoRegressive(
            value=RandNormal(),
            coefficient=RandUniform(
                low=-1.0,
                high=1.0,
            ),
            noise=RandNormal(),
            order=RandInt(low=1, high=6),
            max_abs_value=100.0,
            warmup=warmup,
        )._warmup
        == warmup
    )


def test_auto_regressive_warmup_default() -> None:
    assert (
        AutoRegressive(
            value=RandNormal(),
            coefficient=RandUniform(
                low=-1.0,
                high=1.0,
            ),
            noise=RandNormal(),
            order=RandInt(low=1, high=6),
            max_abs_value=100.0,
        )._warmup
        == 10
    )


@pytest.mark.parametrize("warmup", [-10, -1])
def test_auto_regressive_warmup_incorrect(warmup: int) -> None:
    with pytest.raises(ValueError, match="warmup has to be positive or zero"):
        AutoRegressive(
            value=RandNormal(),
            coefficient=RandUniform(
                low=-1.0,
                high=1.0,
            ),
            noise=RandNormal(),
            order=RandInt(low=1, high=6),
            max_abs_value=100.0,
            warmup=warmup,
        )


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
def test_auto_regressive_generate(batch_size: int, seq_len: int, feature_size: int) -> None:
    batch = AutoRegressive(
        value=RandNormal(feature_size=feature_size),
        coefficient=RandUniform(low=-1.0, high=1.0, feature_size=feature_size),
        noise=RandNormal(feature_size=feature_size),
        order=RandInt(low=1, high=6),
        max_abs_value=100.0,
    ).generate(batch_size=batch_size, seq_len=seq_len)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (batch_size, seq_len, feature_size)
    assert batch.dtype == torch.float
    assert batch.max() <= 100.0
    assert batch.min() >= -100.0


def test_auto_regressive_generate_incorrect_order() -> None:
    generator = AutoRegressive(
        value=RandNormal(),
        coefficient=RandUniform(low=-1.0, high=1.0),
        noise=RandNormal(),
        order=RandInt(low=-6, high=1),
        max_abs_value=100.0,
    )
    with pytest.raises(RuntimeError, match="Order must be a positive integer"):
        generator.generate(batch_size=4, seq_len=12)


def test_auto_regressive_generate_same_random_seed() -> None:
    generator = AutoRegressive(
        value=RandNormal(),
        coefficient=RandUniform(low=-1.0, high=1.0),
        noise=Full(0.0),
        order=RandInt(low=1, high=6),
        max_abs_value=100.0,
    )
    assert objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
    )


def test_auto_regressive_generate_different_random_seeds() -> None:
    generator = AutoRegressive(
        value=RandNormal(),
        coefficient=RandUniform(low=-1.0, high=1.0),
        noise=Full(0.0),
        order=RandInt(low=1, high=6),
        max_abs_value=100.0,
    )
    assert not objects_are_equal(
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(1)),
        generator.generate(batch_size=4, seq_len=12, rng=get_torch_generator(2)),
    )
