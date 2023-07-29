from __future__ import annotations

__all__ = ["prepare_probabilities", "prepare_weighted_generators"]

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

GENERATOR = "generator"
WEIGHT = "weight"


def prepare_probabilities(weights: Tensor | Sequence[int | float]) -> torch.Tensor:
    r"""Converts un-normalized positive weights to probabilities.

    Args:
    ----
        weights (``torch.Tensor`` of shape ``(num_categories,)`` and
            type float or ``Sequence``): Specifies the vector of
            weights associated to each category. The weights have
            to be positive.

    Returns:
    -------
        ``torch.Tensor`` of type float and shape ``(num_categories,)``:
            The vector of probability associated at each category.

    Raises:
    ------
        ValueError if the weights are not valid.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.utils.weight import prepare_probabilities
        >>> prepare_probabilities([1, 1, 1, 1])
        tensor([0.2500, 0.2500, 0.2500, 0.2500])
    """
    if not torch.is_tensor(weights):
        weights = torch.as_tensor(weights)
    if weights.ndim != 1:
        raise ValueError(f"weights has to be a 1D tensor (received a {weights.ndim}D tensor)")
    if weights.min() < 0:
        raise ValueError(
            f"The values in weights have to be positive (min: {weights.min()}  weights: {weights})"
        )
    if weights.sum() == 0:
        raise ValueError(
            f"The sum of the weights has to be greater than 0 (sum: {weights.sum()}  "
            f"weights: {weights})"
        )
    return weights.float() / weights.sum()


def prepare_weighted_generators(
    generators: Sequence[dict],
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    r"""Prepare the tensor generators.

    Each dictionary in the input tuple/list should have the
    following items:

        - a key ``'generator'`` which indicates the tensor generator
            or its configuration.
        - an optional key ``'weight'`` with a float value which
            indicates the weight of the tensor generator.
            If this key is absent, the weight is set to ``1.0``.

    Args:
    ----
        generators (tuple or list): Specifies the tensor generators
            and their weights. See above to learn about the
            expected format.

    Returns:
    -------
        tuple with two values:
            - a tuple of generators or their configurations
            - a tuple of floats

    Example usage:

    .. code-block:: pycon

        >>> from startorch.utils.weight import prepare_weighted_generators
        >>> from startorch.tensor import RandUniform, RandNormal
        >>> prepare_weighted_generators(
        ...     (
        ...         {"weight": 2.0, "generator": RandUniform()},
        ...         {"weight": 1.0, "generator": RandNormal()},
        ...     )
        ... )
        ((RandUniformTensorGenerator(low=0.0, high=1.0), RandNormalTensorGenerator(mean=0.0, std=1.0)), (2.0, 1.0))
    """
    gens = []
    weights = []
    for generator in generators:
        gens.append(generator[GENERATOR])
        weights.append(float(generator.get(WEIGHT, 1.0)))
    return tuple(gens), tuple(weights)
