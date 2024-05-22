r"""Contain an example generator to generate regression data using the
sparse uncorrelated features."""

from __future__ import annotations

__all__ = ["make_sparse_uncorrelated_regression"]


import torch

from startorch import constants as ct
from startorch.random import normal, rand_normal
from startorch.utils.validation import check_feature_size, check_num_examples, check_std


def make_sparse_uncorrelated_regression(
    num_examples: int = 100,
    feature_size: int = 4,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    r"""Generate a random regression problem with sparse uncorrelated
    design.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_uncorrelated.html

    Args:
        num_examples: The number of examples.
        feature_size: The feature size. The feature size has
            to be greater than or equal to 4. Out of all features,
            only 4 are actually used to compute the targets.
            The remaining features are independent of targets.
        noise_std: The standard deviation of the Gaussian
            noise.
        generator: An optional random generator.

    Returns:
        A batch with two items:
            - ``'input'``: a ``BatchedTensor`` of type float and
                shape ``(num_examples, feature_size)``. This
                tensor represents the input features.
            - ``'target'``: a ``BatchedTensor`` of type float and
                shape ``(num_examples,)``. This tensor represents
                the targets.

    Raises:
        RuntimeError: if one of the parameters is not valid.

    Example usage:

    ```pycon

    >>> from startorch.example import make_sparse_uncorrelated_regression
    >>> data = make_sparse_uncorrelated_regression(num_examples=10)
    >>> data
    {'target': tensor([...]), 'feature': tensor([[...]])}

    ```
    """
    check_num_examples(num_examples)
    check_feature_size(feature_size, low=4)
    check_std(noise_std, "noise_std")

    features = rand_normal(size=(num_examples, feature_size), generator=generator)
    targets = normal(
        mean=(features[:, 0] + 2 * features[:, 1] - 2 * features[:, 2] - 1.5 * features[:, 3]),
        std=torch.ones(num_examples),
        generator=generator,
    )
    return {ct.TARGET: targets, ct.FEATURE: features}
