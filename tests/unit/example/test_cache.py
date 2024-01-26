from __future__ import annotations

import torch
from pytest import mark
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example import Cache, SwissRoll
from startorch.utils.seed import get_torch_generator

SIZES = [1, 2, 4]


###########################################
#     Tests for CacheExampleGenerator     #
###########################################


def test_cache_str() -> None:
    assert str(Cache(SwissRoll())).startswith("CacheExampleGenerator(")


@mark.parametrize("batch_size", SIZES)
def test_cache_generate(batch_size: int) -> None:
    data = Cache(SwissRoll()).generate(batch_size)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], BatchedTensor)
    assert data[ct.TARGET].batch_size == batch_size
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.float
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert data[ct.FEATURE].batch_size == batch_size
    assert data[ct.FEATURE].shape == (batch_size, 3)
    assert data[ct.FEATURE].dtype == torch.float


def test_cache_generate_same_batch() -> None:
    generator = Cache(SwissRoll())
    batch1 = generator.generate(8)
    batch2 = generator.generate(8)
    assert batch1.equal(batch2)
    assert batch1 is batch2


def test_cache_generate_copied_batch() -> None:
    generator = Cache(SwissRoll(), deepcopy=True)
    batch1 = generator.generate(8)
    batch2 = generator.generate(8)
    assert batch1.equal(batch2)
    assert batch1 is not batch2


def test_cache_generate_different_batch_size() -> None:
    generator = Cache(SwissRoll(), deepcopy=True)
    batch1 = generator.generate(8)
    batch2 = generator.generate(6)
    assert batch1.batch_size == 8
    assert batch2.batch_size == 6


def test_cache_generate_same_random_seed() -> None:
    generator = Cache(SwissRoll())
    assert generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1))
    )


def test_cache_generate_different_random_seeds() -> None:
    generator = Cache(SwissRoll())
    # The batches are the same because the batch was cached
    assert generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(2))
    )
