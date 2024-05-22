r"""Contain utility functions to prepare weighted generators."""

from __future__ import annotations

__all__ = ["prepare_probabilities", "prepare_weighted_generators"]


from typing import TYPE_CHECKING, Any

from startorch.utils.conversion import to_tensor

GENERATOR = "generator"
WEIGHT = "weight"


if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


def prepare_probabilities(weights: torch.Tensor | Sequence[float]) -> torch.Tensor:
    r"""Convert un-normalized positive weights to probabilities.

    Args:
        weights: The vector of weights associated to each
            category. The weights have to be positive. It must be a
            float tensor of shape ``(num_categories,)`` or a
            ``Sequence``.

    Returns:
        The vector of probability associated at each category.
            The output is a ``torch.Tensor`` of type float and
            shape ``(num_categories,)``

    Raises:
        ValueError: if the weights are not valid.

    Example usage:

    ```pycon

    >>> from startorch.utils.weight import prepare_probabilities
    >>> prepare_probabilities([1, 1, 1, 1])
    tensor([0.2500, 0.2500, 0.2500, 0.2500])

    ```
    """
    weights = to_tensor(weights)
    if weights.ndim != 1:
        msg = f"weights has to be a 1D tensor (received a {weights.ndim}D tensor)"
        raise ValueError(msg)
    if weights.min() < 0:
        msg = (
            f"The values in weights have to be positive (min: {weights.min()}  weights: {weights})"
        )
        raise ValueError(msg)
    if weights.sum() == 0:
        msg = (
            f"The sum of the weights has to be greater than 0 (sum: {weights.sum()}  "
            f"weights: {weights})"
        )
        raise ValueError(msg)
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
        generators: The tensor generators and their weights.
            See above to learn about the expected format.

    Returns:
        A tuple with two items:
            - a tuple of generators or their configurations
            - a tuple of generator weights

    Example usage:

    ```pycon

    >>> from startorch.utils.weight import prepare_weighted_generators
    >>> from startorch.tensor import RandUniform, RandNormal
    >>> prepare_weighted_generators(
    ...     (
    ...         {"weight": 2.0, "generator": RandUniform()},
    ...         {"weight": 1.0, "generator": RandNormal()},
    ...     )
    ... )
    ((RandUniformTensorGenerator(low=0.0, high=1.0), RandNormalTensorGenerator(mean=0.0, std=1.0)), (2.0, 1.0))

    ```
    """
    gens = []
    weights = []
    for generator in generators:
        gens.append(generator[GENERATOR])
        weights.append(float(generator.get(WEIGHT, 1.0)))
    return tuple(gens), tuple(weights)
