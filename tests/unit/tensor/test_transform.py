from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.tensor import Full, RandNormal, TransformTensorGenerator
from startorch.tensor.transformer import Abs
from startorch.utils.seed import get_torch_generator

SIZES = ((1,), (2, 3), (2, 3, 4))


##############################################
#     Tests for TransformTensorGenerator     #
##############################################


def test_transform_str() -> None:
    assert str(TransformTensorGenerator(generator=RandNormal(), transformer=Abs())).startswith(
        "TransformTensorGenerator("
    )


@pytest.mark.parametrize("size", SIZES)
def test_transform_generate(size: tuple[int, ...]) -> None:
    generator = TransformTensorGenerator(generator=RandNormal(), transformer=Abs())
    tensor = generator.generate(size)
    assert tensor.shape == size
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0


def test_transform_generate_fixed_value() -> None:
    assert (
        TransformTensorGenerator(generator=Full(-1), transformer=Abs())
        .generate(size=(2, 4))
        .equal(torch.ones(2, 4))
    )


def test_transform_generate_same_random_seed() -> None:
    generator = TransformTensorGenerator(generator=RandNormal(), transformer=Abs())
    assert objects_are_equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1)),
        generator.generate(size=(4, 12), rng=get_torch_generator(1)),
    )


def test_transform_generate_different_random_seeds() -> None:
    generator = TransformTensorGenerator(generator=RandNormal(), transformer=Abs())
    assert not objects_are_equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1)),
        generator.generate(size=(4, 12), rng=get_torch_generator(2)),
    )
