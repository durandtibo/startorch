from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET

from startorch.tensor import BaseTensorGenerator, RandUniform
from startorch.transition import TensorTransitionGenerator
from startorch.utils.seed import get_torch_generator

###############################################
#     Tests for TensorTransitionGenerator     #
###############################################


def test_tensor_str() -> None:
    assert str(TensorTransitionGenerator(generator=RandUniform())).startswith(
        "TensorTransitionGenerator("
    )


@pytest.mark.parametrize(
    "generator", [RandUniform(), {OBJECT_TARGET: "startorch.tensor.RandUniform"}]
)
@pytest.mark.parametrize("n", [1, 2, 6])
def test_tensor_generate(generator: BaseTensorGenerator | dict, n: int) -> None:
    out = TensorTransitionGenerator(generator).generate(n=n)
    assert out.shape == (n, n)
    assert out.dtype == torch.float


def test_tensor_generate_same_random_seed() -> None:
    generator = TensorTransitionGenerator(generator=RandUniform())
    assert objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(1)),
    )


def test_tensor_generate_different_random_seeds() -> None:
    generator = TensorTransitionGenerator(generator=RandUniform())
    assert not objects_are_equal(
        generator.generate(n=9, rng=get_torch_generator(1)),
        generator.generate(n=9, rng=get_torch_generator(2)),
    )
