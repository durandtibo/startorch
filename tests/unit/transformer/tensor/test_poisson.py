from __future__ import annotations

import torch

from startorch.transformer.tensor import Poisson
from startorch.utils.seed import get_torch_generator

#############################
#     Tests for Poisson     #
#############################


def test_poisson_str() -> None:
    assert str(Poisson()).startswith("PoissonTensorTransformer(")


def test_poisson_transform() -> None:
    tensor = Poisson().transform([torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])])
    assert tensor.shape == (2, 3)
    assert tensor.dtype == torch.float


def test_poisson_transform_same_random_seed() -> None:
    transformer = Poisson()
    rate = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert transformer.transform([rate], rng=get_torch_generator(1)).equal(
        transformer.transform([rate], rng=get_torch_generator(1))
    )


def test_poisson_transform_different_random_seeds() -> None:
    transformer = Poisson()
    rate = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert not transformer.transform([rate], rng=get_torch_generator(1)).equal(
        transformer.transform([rate], rng=get_torch_generator(2))
    )
