from __future__ import annotations

import torch
from coola import objects_are_equal
from pytest import fixture, mark, raises
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example import make_blobs_classification
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


@fixture
def centers() -> torch.Tensor:
    return torch.rand(5, 2) * 20.0 - 10.0


###############################################
#     Tests for make_blobs_classification     #
###############################################


@mark.parametrize("num_examples", (0, -1))
def test_make_blobs_classification_incorrect_num_examples(
    num_examples: int, centers: torch.Tensor
) -> None:
    with raises(
        RuntimeError,
        match="Incorrect value for num_examples. Expected a value greater or equal to 1",
    ):
        make_blobs_classification(num_examples=num_examples, centers=centers)


def test_make_blobs_classification_incorrect_centers_cluster_std(centers: torch.Tensor) -> None:
    with raises(RuntimeError, match="centers and cluster_std do not match:"):
        make_blobs_classification(num_examples=4, centers=centers, cluster_std=torch.ones(2, 2))


def test_make_blobs_classification(centers: torch.Tensor) -> None:
    data = make_blobs_classification(num_examples=100, centers=centers)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    targets = data[ct.TARGET]
    assert isinstance(targets, BatchedTensor)
    assert targets.batch_size == 100
    assert targets.shape == (100,)
    assert targets.dtype == torch.long
    assert targets.min() >= 0
    assert targets.max() <= 4

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert features.batch_size == 100
    assert features.shape == (100, 2)
    assert features.dtype == torch.float


def test_make_blobs_classification_cluster_std_tensor(centers: torch.Tensor) -> None:
    data = make_blobs_classification(
        num_examples=100, centers=centers, cluster_std=torch.ones_like(centers)
    )
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    targets = data[ct.TARGET]
    assert isinstance(targets, BatchedTensor)
    assert targets.batch_size == 100
    assert targets.shape == (100,)
    assert targets.dtype == torch.long
    assert targets.min() >= 0
    assert targets.max() <= 4

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert features.batch_size == 100
    assert features.shape == (100, 2)
    assert features.dtype == torch.float


@mark.parametrize("num_examples", SIZES)
def test_make_blobs_classification_num_examples(num_examples: int, centers: torch.Tensor) -> None:
    data = make_blobs_classification(
        num_examples,
        centers=centers,
    )
    assert len(data) == 2
    assert data[ct.TARGET].batch_size == num_examples
    assert data[ct.FEATURE].batch_size == num_examples


@mark.parametrize("num_centers", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_make_blobs_classification_num_centers(num_centers: int, feature_size: int) -> None:
    data = make_blobs_classification(
        num_examples=64,
        centers=torch.rand(num_centers, feature_size),
    )
    assert len(data) == 2
    targets = data[ct.TARGET]
    assert isinstance(targets, BatchedTensor)
    assert targets.batch_size == 64
    assert targets.shape == (64,)
    assert targets.dtype == torch.long
    assert targets.min() >= 0
    assert targets.max() <= num_centers - 1

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert features.batch_size == 64
    assert features.shape == (64, feature_size)
    assert features.dtype == torch.float


def test_make_blobs_classification_1_center() -> None:
    data = make_blobs_classification(num_examples=64, centers=torch.rand(1, 4))
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    assert data[ct.TARGET].equal(BatchedTensor(torch.zeros(64, dtype=torch.long)))

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert features.batch_size == 64
    assert features.shape == (64, 4)
    assert features.dtype == torch.float


def test_make_blobs_classification_same_random_seed(centers: torch.Tensor) -> None:
    assert objects_are_equal(
        make_blobs_classification(
            num_examples=64,
            centers=centers,
            generator=get_torch_generator(1),
        ),
        make_blobs_classification(
            num_examples=64,
            centers=centers,
            generator=get_torch_generator(1),
        ),
    )


def test_make_blobs_classification_different_random_seeds(centers: torch.Tensor) -> None:
    assert not objects_are_equal(
        make_blobs_classification(
            num_examples=64,
            centers=centers,
            generator=get_torch_generator(1),
        ),
        make_blobs_classification(
            num_examples=64,
            centers=centers,
            generator=get_torch_generator(2),
        ),
    )
