from __future__ import annotations

__all__ = ["make_friedman1", "make_friedman2", "make_friedman3"]

import math

import torch
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.random import rand_normal, rand_uniform


def make_friedman1(
    num_examples: int = 100,
    feature_size: int = 10,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> BatchDict[BatchedTensor]:
    r"""Generates the "Friedman #1" regression problem.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html

    Args:
    ----
        num_examples (int, optional): Specifies the number of examples.
            Default: ``100``
        feature_size (int, optional): Specifies the feature size.
            The feature size has to be greater than or equal to 5.
            Out of all features, only 5 are actually used to compute
            the targets. The remaining features are independent of
            targets. Default: ``10``
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

        >>> from startorch.example import make_friedman1
        >>> batch = make_friedman1(num_examples=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """
    if num_examples < 1:
        raise RuntimeError(f"The number of examples ({num_examples}) has to be greater than 0")
    if feature_size < 5:
        raise RuntimeError(f"feature_size ({feature_size}) has to be greater or equal to 5")
    if noise_std < 0:
        raise RuntimeError(
            f"The standard deviation of the Gaussian noise ({noise_std}) has to be "
            "greater or equal than 0"
        )

    features = rand_uniform(size=(num_examples, feature_size), generator=generator)
    targets = (
        10 * torch.sin(math.pi * features[:, 0] * features[:, 1])
        + 20 * (features[:, 2] - 0.5) ** 2
        + 10 * features[:, 3]
        + 5 * features[:, 4]
    )
    if noise_std > 0.0:
        targets += rand_normal(size=(num_examples,), std=noise_std, generator=generator)
    return BatchDict({ct.TARGET: BatchedTensor(targets), ct.FEATURE: BatchedTensor(features)})


def make_friedman2(
    num_examples: int = 100,
    feature_size: int = 4,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> BatchDict[BatchedTensor]:
    r"""Generates the "Friedman #2" regression problem.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman2.html

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

        >>> from startorch.example import make_friedman2
        >>> batch = make_friedman2(num_examples=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """
    if num_examples < 1:
        raise RuntimeError(f"The number of examples ({num_examples}) has to be greater than 0")
    if feature_size < 4:
        raise RuntimeError(f"feature_size ({feature_size}) has to be greater or equal to 4")
    if noise_std < 0:
        raise RuntimeError(
            f"The standard deviation of the Gaussian noise ({noise_std}) has to be "
            "greater or equal than 0"
        )

    features = rand_uniform(size=(num_examples, feature_size), generator=generator)
    features[:, 0] *= 100
    features[:, 1] *= 520 * math.pi
    features[:, 1] += 40 * math.pi
    features[:, 3] *= 10
    features[:, 3] += 1

    targets = (
        features[:, 0] ** 2
        + (features[:, 1] * features[:, 2] - 1 / (features[:, 1] * features[:, 3])) ** 2
    ) ** 0.5
    if noise_std > 0.0:
        targets += rand_normal(size=(num_examples,), std=noise_std, generator=generator)
    return BatchDict({ct.TARGET: BatchedTensor(targets), ct.FEATURE: BatchedTensor(features)})


def make_friedman3(
    num_examples: int = 100,
    feature_size: int = 4,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> BatchDict[BatchedTensor]:
    r"""Generates the "Friedman #3" regression problem.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman3.html

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

        >>> from startorch.example import make_friedman3
        >>> batch = make_friedman3(num_examples=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """
    if num_examples < 1:
        raise RuntimeError(f"The number of examples ({num_examples}) has to be greater than 0")
    if feature_size < 4:
        raise RuntimeError(f"feature_size ({feature_size}) has to be greater or equal to 4")
    if noise_std < 0:
        raise RuntimeError(
            f"The standard deviation of the Gaussian noise ({noise_std}) has to be "
            "greater or equal than 0"
        )

    features = rand_uniform(size=(num_examples, feature_size), generator=generator)
    features[:, 0] *= 100
    features[:, 1] *= 520 * math.pi
    features[:, 1] += 40 * math.pi
    features[:, 3] *= 10
    features[:, 3] += 1

    targets = torch.atan(
        (features[:, 1] * features[:, 2] - 1 / (features[:, 1] * features[:, 3])) / features[:, 0]
    )
    if noise_std > 0.0:
        targets += rand_normal(size=(num_examples,), std=noise_std, generator=generator)
    return BatchDict({ct.TARGET: BatchedTensor(targets), ct.FEATURE: BatchedTensor(features)})
