r"""This module implements functions to sample values from discrete
univariate distribution supported on a bounded interval."""

from __future__ import annotations

__all__ = ["rand_poisson"]


import torch
from torch import Tensor


def rand_poisson(
    size: list[int] | tuple[int, ...],
    rate: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Creates a tensor filled with values sampled from a Poisson
    distribution.

    Args:
    ----
        size (list or tuple): Specifies the tensor shape.
        rate (float, optional): Specifies the rate of the Poisson
            distribution. This value has to be greater than 0.
            Default: ``1.0``
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor`` of type float: A tensor filled with values
            sampled from a Poisson distribution.

    Raises:
    ------
        ValueError if the ``rate`` parameter is not valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import rand_poisson
        >>> rand_poisson(size=(2, 3), rate=2.0)
        tensor([...])
    """
    if rate <= 0:
        raise ValueError(f"rate has to be greater than 0 (received: {rate})")
    return torch.poisson(torch.full(size, rate, dtype=torch.float), generator=generator)
