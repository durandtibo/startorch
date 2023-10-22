from __future__ import annotations

__all__ = ["make_blobs_classification"]

import math

import torch
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example.utils import check_num_examples
from startorch.random import normal


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
            ``(num_centers, feature_size)``): Specifies the cluster
            centers used to generate the examples.
        cluster_std (``torch.Tensor`` of type float and shape
            ``(num_centers, feature_size)`` or int or float):
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
