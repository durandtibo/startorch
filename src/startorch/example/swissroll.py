from __future__ import annotations

__all__ = ["SwissRollExampleGenerator", "make_swiss_roll"]

import math

import torch
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example.base import BaseExampleGenerator
from startorch.example.utils import check_interval, check_num_examples, check_std
from startorch.random import rand_normal, rand_uniform


class SwissRollExampleGenerator(BaseExampleGenerator[BatchedTensor]):
    r"""Implements a manifold example generator based on the Swiss roll
    pattern.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html

    Args:
    ----
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise. Default: ``0.0``
        spin (float or int, optional): Specifies the number of spins
            of the Swiss roll. Default: ``1.5``
        hole (bool, optional): If ``True`` generates the Swiss roll
            with hole dataset. Default: ``False``

    Raises:
    ------
        ValueError if one of the parameters is not valid.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import SwissRoll
        >>> generator = SwissRoll()
        >>> generator
        SwissRollExampleGenerator(noise_std=0.0, spin=1.5, hole=False)
        >>> batch = generator.generate(batch_size=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        spin: float = 1.5,
        hole: bool = False,
    ) -> None:
        check_std(noise_std, "noise_std")
        self._noise_std = float(noise_std)

        check_interval(spin, low=1e-10, high=math.inf, name="spin")
        self._spin = float(spin)
        self._hole = bool(hole)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(noise_std={self._noise_std:,}, "
            f"spin={self._spin:,}, hole={self._hole})"
        )

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
            hole=self._hole,
            generator=rng,
        )


def make_swiss_roll(
    num_examples: int = 100,
    noise_std: float = 0.0,
    spin: float = 1.5,
    hole: bool = False,
    generator: torch.Generator | None = None,
) -> BatchDict[BatchedTensor]:
    r"""Generates a toy manifold dataset based on Swiss roll pattern.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html

    Args:
    ----
        num_examples (int, optional): Specifies the number of examples.
            Default: ``100``
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise. Default: ``0.0``
        spin (float or int, optional): Specifies the number of spins
            of the Swiss roll. Default: ``1.5``
        hole (bool, optional): If ``True`` generates the Swiss roll
            with hole dataset. Default: ``False``
        generator (``torch.Generator`` or ``None``, optional):
            Specifies an optional random generator.

    Returns:
    -------
        ``BatchDict``: A batch with two items:
            - ``'input'``: a ``BatchedTensor`` of type float and
                shape ``(num_examples, 3)``. This
                tensor represents the input features.
            - ``'target'``: a ``BatchedTensor`` of type float and
                shape ``(num_examples,)``. This tensor represents
                the targets.

    Raises:
    ------
        RuntimeError if one of the parameters is not valid.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import make_swiss_roll
        >>> batch = make_swiss_roll(num_examples=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """
    check_num_examples(num_examples)
    check_std(noise_std, "noise_std")
    check_interval(spin, low=1e-10, high=math.inf, name="spin")
    if hole:
        corners = torch.tensor(
            [[math.pi * (1.5 + i), j * 7] for i in range(3) for j in range(3) if (i != 1 or j != 1)]
        )
        corner_index = torch.multinomial(
            torch.ones(8), num_examples, replacement=True, generator=generator
        )
        parameters = rand_uniform(size=(num_examples, 2), generator=generator) * torch.tensor(
            [[math.pi, 7.0]]
        )
        values = corners[corner_index] + parameters
        targets, y = values[:, 0:1], values[:, 1:2]
    else:
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
