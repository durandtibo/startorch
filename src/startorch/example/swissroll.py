from __future__ import annotations

__all__ = ["SwissRollExampleGenerator", "make_swiss_roll"]

import math

import torch
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example.base import BaseExampleGenerator
from startorch.random import rand_normal, rand_uniform


class SwissRollExampleGenerator(BaseExampleGenerator[BatchedTensor]):
    r"""Implements a regression example generator based on the Swiss roll
    pattern.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html

    Args:
    ----
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise. Default: ``0.0``
        spin (float or int, optional): Specifies the number of spins
            of the Swiss roll. Default: ``1.5``

    Raises:
    ------
        ValueError if one of the parameters is not valid.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import Hypercube
        >>> generator = Hypercube(num_classes=5, feature_size=6)
        >>> generator
        HypercubeExampleGenerator(num_classes=5, feature_size=6, noise_std=0.2)
        >>> batch = generator.generate(batch_size=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """

    def __init__(self, noise_std: float = 0.0, spin: float | int = 1.5) -> None:
        if noise_std < 0.0:
            raise ValueError(
                f"The standard deviation of the Gaussian noise ({noise_std}) has to be "
                "greater or equal than 0"
            )
        self._noise_std = float(noise_std)

        if spin <= 0.0:
            raise ValueError(
                f"The spin of the Swiss roll ({spin}) has to be greater or equal than 0"
            )
        self._spin = float(spin)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(noise_std={self._noise_std:,}, spin={self._spin:,})"

    @property
    def noise_std(self) -> float:
        r"""``float``: The standard deviation of the Gaussian noise."""
        return self._noise_std

    @property
    def spin(self) -> float:
        r"""``float``: The number of spins."""
        return self._spin

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> BatchDict[BatchedTensor]:
        return make_swiss_roll(
            num_examples=batch_size,
            noise_std=self._noise_std,
            spin=self._spin,
            generator=rng,
        )


def make_swiss_roll(
    num_examples: int = 1000,
    noise_std: float = 0.0,
    spin: float | int = 1.5,
    generator: torch.Generator | None = None,
) -> BatchDict[BatchedTensor]:
    r"""Generates a toy classification dataset based on Swiss roll
    pattern.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html

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
