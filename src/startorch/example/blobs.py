r"""Contain an example generator to generate binary classification
example generator where the data are generated from isotropic Gaussian
blobs."""

from __future__ import annotations

__all__ = ["BlobsClassificationExampleGenerator", "make_blobs_classification"]

import math

import torch
from batchtensor.nested import shuffle_along_batch, slice_along_batch

from startorch import constants as ct
from startorch.example.base import BaseExampleGenerator
from startorch.random import normal
from startorch.utils.seed import get_torch_generator
from startorch.utils.validation import check_num_examples


class BlobsClassificationExampleGenerator(BaseExampleGenerator):
    r"""Implement a binary classification example generator where the
    data are generated from isotropic Gaussian blobs.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

    Args:
        centers: The cluster centers used to generate the
            examples. It must be a float tensor of shape
            ``(num_clusters, feature_size)``.
        cluster_std: The standard deviation of the clusters.
            It must be a float tensor of shape
            ``(num_clusters, feature_size)``.

    Raises:
        TypeError: if one of the parameters has an invalid type.
        RuntimeError: if one of the parameters is not valid.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.example import BlobsClassification
    >>> generator = BlobsClassification(torch.rand(5, 4))
    >>> generator
    BlobsClassificationExampleGenerator(num_clusters=5, feature_size=4)
    >>> batch = generator.generate(batch_size=10)
    >>> batch
    {'target': tensor([...]), 'feature': tensor([[...]])}

    ```
    """

    def __init__(self, centers: torch.Tensor, cluster_std: torch.Tensor | float = 1.0) -> None:
        self._centers = centers
        if not torch.is_tensor(cluster_std):
            cluster_std = torch.full_like(centers, cluster_std)
        self._cluster_std = cluster_std

        if self._centers.shape != self._cluster_std.shape:
            msg = (
                f"centers and cluster_std do not match: {self._centers.shape} "
                f"vs {self._cluster_std.shape}"
            )
            raise RuntimeError(msg)

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
        r"""The feature size i.e. the number of features."""
        return self._centers.shape[1]

    @property
    def num_clusters(self) -> int:
        r"""The number of clusters i.e. categories."""
        return self._centers.shape[0]

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        return make_blobs_classification(
            num_examples=batch_size,
            centers=self._centers,
            cluster_std=self._cluster_std,
            generator=rng,
        )

    @classmethod
    def create_uniform_centers(
        cls,
        num_clusters: int = 3,
        feature_size: int = 2,
        random_seed: int = 17532042831661189422,
    ) -> BlobsClassificationExampleGenerator:
        r"""Instantiate a ``BlobsClassificationExampleGenerator`` where
        the centers are sampled from a uniform distribution.

        Args:
            num_clusters: The number of clusters.
            feature_size: The feature size.
            random_seed: The random seed used to generate
                the cluster centers.

        Returns:
            An instantiated example generator.

        Example usage:

        ```pycon
        >>> from startorch.example import BlobsClassification
        >>> generator = BlobsClassification.create_uniform_centers()
        >>> generator
        BlobsClassificationExampleGenerator(num_clusters=3, feature_size=2)
        >>> batch = generator.generate(batch_size=10)
        >>> batch
        {'target': tensor([...]), 'feature': tensor([[...]])}

        ```
        """
        return cls(
            centers=torch.rand(
                num_clusters,
                feature_size,
                generator=get_torch_generator(random_seed),
            )
            .mul(20.0)
            .sub(10.0)
        )


def make_blobs_classification(
    num_examples: int,
    centers: torch.Tensor,
    cluster_std: torch.Tensor | float = 1.0,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    r"""Generate a classification dataset where the data are gnerated
    from isotropic Gaussian blobs for clustering.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

    Args:
        num_examples: The number of examples.
        centers: The cluster centers used to generate the
            examples. It must be a float tensor of shape
            ``(num_clusters, feature_size)``.
        cluster_std: The standard deviation of the clusters.
            It must be a float tensor of shape
            ``(num_clusters, feature_size)``.
        generator: An optional random number generator.

    Returns:
        A dictionary with two items:
            - ``'input'``: a ``BatchedTensor`` of type float and
                shape ``(num_examples, feature_size)``. This
                tensor represents the input features.
            - ``'target'``: a ``BatchedTensor`` of type long and
                shape ``(num_examples,)``. This tensor represents
                the targets.

    Raises:
        RuntimeError: if one of the parameters is not valid.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.example import make_blobs_classification
    >>> batch = make_blobs_classification(num_examples=10, centers=torch.rand(5, 2))
    >>> batch
    {'target': tensor([...]), 'feature': tensor([[...]])}

    ```
    """
    check_num_examples(num_examples)
    num_centers, feature_size = centers.shape
    if not torch.is_tensor(cluster_std):
        cluster_std = torch.full_like(centers, cluster_std)
    if centers.shape != cluster_std.shape:
        msg = f"centers and cluster_std do not match: {centers.shape} vs {cluster_std.shape}"
        raise RuntimeError(msg)
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

    batch = {ct.TARGET: targets, ct.FEATURE: features}
    batch = shuffle_along_batch(batch, generator)
    return slice_along_batch(batch, stop=num_examples)
