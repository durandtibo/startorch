from __future__ import annotations

import torch

from startorch.utils.seed import get_random_seed, get_torch_generator

#####################################
#     Tests for get_random_seed     #
#####################################


def test_get_random_seed() -> None:
    assert isinstance(get_random_seed(42), int)


def test_get_random_seed_same_seed() -> None:
    assert get_random_seed(42) == get_random_seed(42)


def test_get_random_seed_different_seeds() -> None:
    assert get_random_seed(1) != get_random_seed(42)


#########################################
#     Tests for get_torch_generator     #
#########################################


def test_get_torch_generator_same_seed() -> None:
    assert torch.randn(4, 6, generator=get_torch_generator(1)).equal(
        torch.randn(4, 6, generator=get_torch_generator(1))
    )


def test_get_torch_generator_different_seeds() -> None:
    assert not torch.randn(4, 6, generator=get_torch_generator(1)).equal(
        torch.randn(4, 6, generator=get_torch_generator(2))
    )
