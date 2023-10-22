from __future__ import annotations

from unittest.mock import patch

import torch
from coola import objects_are_equal
from pytest import fixture, mark, raises
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example import BlobsClassification, make_blobs_classification
from startorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


@fixture
def centers() -> torch.Tensor:
    return torch.rand(5, 2) * 20.0 - 10.0


#########################################################
#     Tests for BlobsClassificationExampleGenerator     #
#########################################################


def test_blobs_classification_str(centers: torch.Tensor) -> None:
    assert str(BlobsClassification(centers=centers)).startswith(
        "BlobsClassificationExampleGenerator("
    )


def test_blobs_classification_centers(centers: torch.Tensor) -> None:
    assert BlobsClassification(centers=centers).centers.equal(centers)


@mark.parametrize("cluster_std", (0.1, 1, 4.2))
def test_blobs_classification_cluster_std_scalar(centers: torch.Tensor, cluster_std: float) -> None:
    assert BlobsClassification(centers=centers, cluster_std=cluster_std).cluster_std.equal(
        torch.full((5, 2), cluster_std)
    )


def test_blobs_classification_cluster_std_tensor(centers: torch.Tensor) -> None:
    assert BlobsClassification(
        centers=centers,
        cluster_std=torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]),
    ).cluster_std.equal(torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]))


def test_blobs_classification_cluster_std_incorrect_shape(centers: torch.Tensor) -> None:
    with raises(RuntimeError, match="centers and cluster_std do not match:"):
        BlobsClassification(centers=centers, cluster_std=torch.ones(5, 4))


@mark.parametrize("feature_size", SIZES)
def test_blobs_classification_feature_size(feature_size: int) -> None:
    assert BlobsClassification(centers=torch.ones(5, feature_size)).feature_size == feature_size


@mark.parametrize("num_clusters", SIZES)
def test_blobs_classification_num_clusters(num_clusters: int) -> None:
    assert BlobsClassification(centers=torch.ones(num_clusters, 4)).num_clusters == num_clusters


@mark.parametrize("batch_size", SIZES)
def test_blobs_classification_generate(centers: torch.Tensor, batch_size: int) -> None:
    data = BlobsClassification(centers=centers).generate(batch_size)
    assert isinstance(data, BatchDict)
    assert len(data) == 2
    targets = data[ct.TARGET]
    assert isinstance(targets, BatchedTensor)
    assert targets.batch_size == batch_size
    assert targets.shape == (batch_size,)
    assert targets.dtype == torch.long
    assert targets.min() >= 0
    assert targets.max() <= 4

    features = data[ct.FEATURE]
    assert isinstance(data[ct.FEATURE], BatchedTensor)
    assert features.batch_size == batch_size
    assert features.shape == (batch_size, 2)
    assert features.dtype == torch.float


def test_blobs_classification_generate_same_random_seed(centers: torch.Tensor) -> None:
    generator = BlobsClassification(centers=centers)
    assert generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(1))
    )


def test_blobs_classification_generate_different_random_seeds(centers: torch.Tensor) -> None:
    generator = BlobsClassification(centers=centers)
    assert not generator.generate(batch_size=64, rng=get_torch_generator(1)).equal(
        generator.generate(batch_size=64, rng=get_torch_generator(2))
    )


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("centers", (torch.tensor([[1.0]]), torch.tensor([[2.0]])))
@mark.parametrize("cluster_std", (torch.tensor([[1.0]]), torch.tensor([[2.0]])))
@mark.parametrize("rng", (None, get_torch_generator(1)))
def test_blobs_classification_generate_mock(
    batch_size: int,
    rng: torch.Generator | None,
    centers: torch.Tensor,
    cluster_std: torch.Tensor,
) -> None:
    generator = BlobsClassification(centers=centers, cluster_std=cluster_std)
    with patch("startorch.example.blobs.make_blobs_classification") as make_mock:
        generator.generate(batch_size=batch_size, rng=rng)
        make_mock.assert_called_once_with(
            num_examples=batch_size,
            centers=centers,
            cluster_std=cluster_std,
            generator=rng,
        )


@mark.parametrize("num_clusters", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_blobs_classification_create_uniform_weights(num_clusters: int, feature_size: int) -> None:
    generator = BlobsClassification.create_uniform_centers(
        num_clusters=num_clusters, feature_size=feature_size
    )
    assert generator.centers.shape == (num_clusters, feature_size)
    assert generator.cluster_std.equal(torch.ones(num_clusters, feature_size))


def test_blobs_classification_create_uniform_weights_default() -> None:
    generator = BlobsClassification.create_uniform_centers()
    assert generator.centers.shape == (3, 2)
    assert generator.cluster_std.equal(torch.ones(3, 2))


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
