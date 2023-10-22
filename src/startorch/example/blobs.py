from __future__ import annotations

__all__ = ["BlobsClassificationExampleGenerator", "make_blobs_classification"]

import math

import torch
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example.base import BaseExampleGenerator
from startorch.example.utils import check_num_examples
from startorch.random import normal


class BlobsClassificationExampleGenerator(BaseExampleGenerator[BatchedTensor]):
    r"""Implements a binary classification example generator where the
    data are generated with a large circle containing a smaller circle
    in 2d.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

    Args:
    ----
        centers (``torch.Tensor`` of type float and shape
            ``(num_clusters, feature_size)``): Specifies the cluster
            centers used to generate the examples.
        cluster_std (``torch.Tensor`` of type float and shape
            ``(num_clusters, feature_size)`` or int or float):
            Specifies standard deviation of the clusters.
            Default: ``1.0``

    Raises:
    ------
        TypeError or RuntimeError if one of the parameters is not
            valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.example import BlobsClassification
        >>> generator = BlobsClassification(torch.rand(5, 4))
        >>> generator
        BlobsClassificationExampleGenerator(num_clusters=5, feature_size=4)
        >>> batch = generator.generate(batch_size=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """

    def __init__(
        self, centers: torch.Tensor, cluster_std: torch.Tensor | int | float = 1.0
    ) -> None:
        self._centers = centers
        if not torch.is_tensor(cluster_std):
            cluster_std = torch.full_like(centers, cluster_std)
        self._cluster_std = cluster_std

        if self._centers.shape != self._cluster_std.shape:
            raise RuntimeError(
                f"centers and cluster_std do not match: {self._centers.shape} "
                f"vs {self._cluster_std.shape}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_clusters={self.num_clusters:,}, "
            f"feature_size={self.feature_size:,})"
        )

    @property
    def centers(self) -> torch.Tensor:
        r"""``torch.Tensor`` of type float and shape ``(num_clusters,
        feature_size)``: The cluster centers."""
        return self._centers

    @property
    def cluster_std(self) -> torch.Tensor:
        r"""``torch.Tensor`` of type float and shape ``(num_clusters,
        feature_size)``: The standard deviation for each cluster."""
        return self._cluster_std

    @property
    def feature_size(self) -> int:
        r"""int: The feature size i.e. the number of features."""
        return self._centers.shape[1]

    @property
    def num_clusters(self) -> int:
        r"""int: The number of clusters i.e. categories."""
        return self._centers.shape[0]

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> BatchDict[BatchedTensor]:
        return make_blobs_classification(
            num_examples=batch_size,
            centers=self._centers,
            cluster_std=self._cluster_std,
            generator=rng,
        )


def make_blobs_classification(
    num_examples: int,
    centers: torch.Tensor,
    cluster_std: torch.Tensor | int | float = 1.0,
    generator: torch.Generator | None = None,
) -> BatchDict[BatchedTensor]:
    r"""Generates a classification dataset where the data are gnerated
    from isotropic Gaussian blobs for clustering.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

    Args:
    ----
        num_examples (int, optional): Specifies the number of examples.
            Default: ``100``
        centers (``torch.Tensor`` of type float and shape
            ``(num_clusters, feature_size)``): Specifies the cluster
            centers used to generate the examples.
        cluster_std (``torch.Tensor`` of type float and shape
            ``(num_clusters, feature_size)`` or int or float):
            Specifies standard deviation of the clusters.
            Default: ``1.0``
        generator (``torch.Generator`` or ``None``, optional):
            Specifies an optional random generator. Default: ``None``

    Returns:
    -------
        ``BatchDict``: A batch with two items:
            - ``'input'``: a ``BatchedTensor`` of type float and
                shape ``(num_examples, feature_size)``. This
                tensor represents the input features.
            - ``'target'``: a ``BatchedTensor`` of type long and
                shape ``(num_examples,)``. This tensor represents
                the targets.

    Raises:
    ------
        RuntimeError if one of the parameters is not valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.example import make_blobs_classification
        >>> batch = make_blobs_classification(num_examples=10, centers=torch.rand(5, 2))
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """
    check_num_examples(num_examples)
    num_centers, feature_size = centers.shape
    if not torch.is_tensor(cluster_std):
        cluster_std = torch.full_like(centers, cluster_std)
    if centers.shape != cluster_std.shape:
        raise RuntimeError(
            f"centers and cluster_std do not match: {centers.shape} vs {cluster_std.shape}"
        )
    num_examples_per_center = math.ceil(num_examples / num_centers)

    features = torch.empty(num_examples_per_center * num_centers, feature_size, dtype=torch.float)
    targets = torch.empty(num_examples_per_center * num_centers, dtype=torch.long)

    for i in range(num_centers):
        start_idx = i * num_examples_per_center
        end_idx = (i + 1) * num_examples_per_center
        features[start_idx:end_idx] = normal(
            mean=centers[i].view(1, feature_size).expand(num_examples_per_center, feature_size),
            std=cluster_std[i].view(1, feature_size).expand(num_examples_per_center, feature_size),
            generator=generator,
        )
        targets[start_idx:end_idx] = i

    batch = BatchDict({ct.TARGET: BatchedTensor(targets), ct.FEATURE: BatchedTensor(features)})
    batch.shuffle_along_batch_(generator)
    return batch.slice_along_batch(stop=num_examples)
