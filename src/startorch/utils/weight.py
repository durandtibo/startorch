from __future__ import annotations

__all__ = ["prepare_probabilities"]

from collections.abc import Sequence

import torch
from torch import Tensor


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

        >>> import torch
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
