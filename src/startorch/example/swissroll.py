from __future__ import annotations

__all__ = ["make_swiss_roll"]

import math

import torch
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.random import rand_normal, rand_uniform

# class SwissRollExampleGenerator(BaseExampleGenerator[BatchedTensor]):
#     def __init__(self, noise_std: float = 0.0) -> None:
#         if noise_std < 0.0:
#             raise ValueError(
#                 f"The standard deviation of the Gaussian noise ({noise_std}) has to be "
#                 "greater or equal than 0"
#             )
#         self._noise_std = float(noise_std)


def make_swiss_roll(
    num_examples: int = 1000,
    noise_std: float = 0.0,
    spin: float | int = 1.5,
    scale: float | int = 21.0,
    generator: torch.Generator | None = None,
) -> BatchDict[BatchedTensor]:
    r"""Generates a toy classification dataset based on Swiss roll
    pattern.

    The implementation is based on https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html

    Args:
    ----
        num_examples (int, optional): Specifies the number of examples.
            Default: ``1000``
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise. Default: ``0.0``
        spin (float or int, optional): Specifies the number of spins
            of the Swiss roll. Default: ``1.5``
        generator (``torch.Generator`` or ``None``, optional):
            Specifies an optional random generator. Default: ``None``

    Returns:
    -------
        dict: A dictionary with two keys:
            - ``'input'``: a ``torch.Tensor`` of type float and
                shape ``(num_examples, 3)``. This
                tensor represents the input features.
            - ``'target'``: a ``torch.Tensor`` of type float and
                shape ``(num_examples,)``. This tensor represents
                the targets.

    Raises:
    ------
        RuntimeError if one of the parameters is not valid.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example.swissroll import make_swiss_roll
        >>> batch = make_swiss_roll(num_examples=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """
    if num_examples < 1:
        raise RuntimeError(f"The number of examples ({num_examples}) has to be greater than 0")
    if noise_std < 0:
        raise RuntimeError(
            f"The standard deviation of the Gaussian noise ({noise_std}) has to be "
            "greater or equal than 0"
        )
    if spin <= 0.0:
        raise RuntimeError(f"The spin of the Swiss roll ({spin}) has to be greater or equal than 0")
    targets = rand_uniform(size=(num_examples, 1), low=1.0, high=3.0, generator=generator).mul(
        spin * math.pi
    )
    y = rand_uniform(size=(num_examples, 1), low=0.0, high=21.0, generator=generator)

    x = targets.cos().mul(targets)
    z = targets.sin().mul(targets)

    features = torch.cat((x, y, z), dim=1)
    if noise_std > 0.0:
        features += rand_normal(size=(num_examples, 3), std=noise_std, generator=generator)
    return BatchDict(
        {ct.TARGET: BatchedTensor(targets.flatten()), ct.FEATURE: BatchedTensor(features)}
    )
