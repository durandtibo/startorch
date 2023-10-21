from __future__ import annotations

__all__ = ["make_sparse_uncorrelated_regression"]

import torch
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example.utils import check_num_examples
from startorch.random import normal, rand_normal


def make_sparse_uncorrelated_regression(
    num_examples: int = 100,
    feature_size: int = 4,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> BatchDict[BatchedTensor]:
    r"""Generates a random regression problem with sparse uncorrelated
    design.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_uncorrelated.html

    Args:
    ----
        num_examples (int, optional): Specifies the number of examples.
            Default: ``100``
        feature_size (int, optional): Specifies the feature size.
            The feature size has to be greater than or equal to 4.
            Out of all features, only 4 are actually used to compute
            the targets. The remaining features are independent of
            targets. Default: ``4``
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise. Default: ``0.0``
        generator (``torch.Generator`` or ``None``, optional):
            Specifies an optional random generator. Default: ``None``

    Returns:
    -------
        ``BatchDict``: A batch with two items:
            - ``'input'``: a ``BatchedTensor`` of type float and
                shape ``(num_examples, feature_size)``. This
                tensor represents the input features.
            - ``'target'``: a ``BatchedTensor`` of type float and
                shape ``(num_examples,)``. This tensor represents
                the targets.

    Raises:
    ------
        RuntimeError if one of the parameters is not valid.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import make_sparse_uncorrelated_regression
        >>> batch = make_sparse_uncorrelated_regression(num_examples=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """
    check_num_examples(num_examples)
    if feature_size < 4:
        raise RuntimeError(f"feature_size ({feature_size}) has to be greater or equal to 4")
    if noise_std < 0:
        raise RuntimeError(
            f"The standard deviation of the Gaussian noise ({noise_std}) has to be "
            "greater or equal than 0"
        )

    features = rand_normal(size=(num_examples, feature_size), generator=generator)
    targets = normal(
        mean=(features[:, 0] + 2 * features[:, 1] - 2 * features[:, 2] - 1.5 * features[:, 3]),
        std=torch.ones(num_examples),
        generator=generator,
    )
    return BatchDict({ct.TARGET: BatchedTensor(targets), ct.FEATURE: BatchedTensor(features)})
