from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET

from startorch.tensor import Full, RandUniform
from startorch.transition import (
    BaseTransitionGenerator,
    NormalizeTransitionGenerator,
    TensorTransitionGenerator,
)
from startorch.utils.seed import get_torch_generator

##################################################
#     Tests for NormalizeTransitionGenerator     #
##################################################


def test_normalize_str() -> None:
    assert str(
        NormalizeTransitionGenerator(generator=TensorTransitionGenerator(Full(1.0)))
    ).startswith("NormalizeTransitionGenerator(")


@pytest.mark.parametrize(
    "generator",
    [
        TensorTransitionGenerator(Full(1.0)),
        {
            OBJECT_TARGET: "TensorTransitionGenerator",
            "generator": {OBJECT_TARGET: "startorch.tensor.Full", "value": 1.0},
        },
    ],
)
@pytest.mark.parametrize("n", [1, 2, 6])
def test_normalize_generate(generator: BaseTransitionGenerator | dict, n: int) -> None:
    out = NormalizeTransitionGenerator(generator).generate(n=n)
    assert out.shape == (n, n)
    assert out.dtype == torch.float


def test_normalize_generate_p_1() -> None:
    out = NormalizeTransitionGenerator(TensorTransitionGenerator(Full(1.0)), p=1.0).generate(n=4)
    assert objects_are_equal(out, torch.full((4, 4), 0.25))


def test_normalize_generate_p_2() -> None:
    out = NormalizeTransitionGenerator(TensorTransitionGenerator(Full(1.0)), p=2.0).generate(n=4)
    assert objects_are_equal(out, torch.full((4, 4), 0.5))


def test_normalize_generate_same_random_seed() -> None:
    generator = NormalizeTransitionGenerator(generator=TensorTransitionGenerator(RandUniform()))
    assert objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(1)),
    )


def test_normalize_generate_different_random_seeds() -> None:
    generator = NormalizeTransitionGenerator(generator=TensorTransitionGenerator(RandUniform()))
    assert not objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(2)),
    )
