from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch import constants as ct
from startorch.example import Cache, SwissRoll
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


###########################################
#     Tests for CacheExampleGenerator     #
###########################################


def test_cache_str() -> None:
    assert str(Cache(SwissRoll())).startswith("CacheExampleGenerator(")


@pytest.mark.parametrize("batch_size", SIZES)
def test_cache_generate(batch_size: int) -> None:
    batch = Cache(SwissRoll()).generate(batch_size)
    assert isinstance(batch, dict)
    assert len(batch) == 2
    assert isinstance(batch[ct.TARGET], torch.Tensor)
    assert batch[ct.TARGET].shape == (batch_size,)
    assert batch[ct.TARGET].dtype == torch.float
    assert isinstance(batch[ct.FEATURE], torch.Tensor)
    assert batch[ct.FEATURE].shape == (batch_size, 3)
    assert batch[ct.FEATURE].dtype == torch.float


def test_cache_generate_same_batch() -> None:
    generator = Cache(SwissRoll())
    batch1 = generator.generate(8)
    batch2 = generator.generate(8)
    assert objects_are_equal(batch1, batch2)
    assert batch1 is batch2


def test_cache_generate_copied_batch() -> None:
    generator = Cache(SwissRoll(), deepcopy=True)
    batch1 = generator.generate(8)
    batch2 = generator.generate(8)
    assert objects_are_equal(batch1, batch2)
    assert batch1 is not batch2


def test_cache_generate_different_batch_size() -> None:
    generator = Cache(SwissRoll(), deepcopy=True)
    batch1 = generator.generate(8)
    batch2 = generator.generate(6)
    assert batch1[ct.TARGET].shape[0] == 8
    assert batch2[ct.TARGET].shape[0] == 6


def test_cache_generate_same_random_seed() -> None:
    generator = Cache(SwissRoll())
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
    )


def test_cache_generate_different_random_seeds() -> None:
    generator = Cache(SwissRoll())
    # The batches are the same because the batch was cached
    assert objects_are_equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1)),
        generator.generate(batch_size=64, rng=get_torch_generator(2)),
    )
