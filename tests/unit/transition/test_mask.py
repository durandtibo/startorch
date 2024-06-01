from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET

from startorch.tensor import Full, RandUniform
from startorch.transition import (
    BaseTransitionGenerator,
    MaskTransitionGenerator,
    TensorTransitionGenerator,
)
from startorch.utils.seed import get_torch_generator

#############################################
#     Tests for MaskTransitionGenerator     #
#############################################


def test_mask_transition_generator_str() -> None:
    assert str(
        MaskTransitionGenerator(generator=TensorTransitionGenerator(Full(1.0)), num_mask=4)
    ).startswith("MaskTransitionGenerator(")


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
def test_mask_transition_generator_generate(
    generator: BaseTransitionGenerator | dict, n: int
) -> None:
    out = MaskTransitionGenerator(generator, num_mask=n - 1).generate(n=n)
    assert out.shape == (n, n)
    assert out.dtype == torch.float
    assert objects_are_equal(out.sum(dim=0), torch.ones(n))
    assert objects_are_equal(out.sum(dim=1), torch.ones(n))


def test_mask_transition_generator_generate_same_random_seed() -> None:
    generator = MaskTransitionGenerator(
        generator=TensorTransitionGenerator(RandUniform()), num_mask=5
    )
    assert objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(1)),
    )


def test_mask_transition_generator_generate_different_random_seeds() -> None:
    generator = MaskTransitionGenerator(
        generator=TensorTransitionGenerator(RandUniform()), num_mask=5
    )
    assert not objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(2)),
    )
