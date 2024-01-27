from __future__ import annotations

__all__ = ["MoonsClassificationExampleGenerator", "make_moons_classification"]

import math

import torch
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example.base import BaseExampleGenerator
from startorch.example.utils import check_interval, check_num_examples, check_std
from startorch.random import rand_normal


class MoonsClassificationExampleGenerator(BaseExampleGenerator[BatchedTensor]):
    r"""Implements a binary classification example generator where the
    data are generated with a large circle containing a smaller circle
    in 2d.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html

    Args:
    ----
        shuffle (bool, optional): If ``True``, the examples are
            shuffled. Default: ``True``
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise.
        ratio (float, optional): Specifies the ratio between the
            number of examples in outer circle and inner circle.
            Default: ``0.5``

    Raises:
    ------
        TypeError or RuntimeError if one of the parameters is not
            valid.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import MoonsClassification
        >>> generator = MoonsClassification()
        >>> generator
        MoonsClassificationExampleGenerator(shuffle=True, noise_std=0.0, ratio=0.5)
        >>> batch = generator.generate(batch_size=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """

    def __init__(self, shuffle: bool = True, noise_std: float = 0.0, ratio: float = 0.5) -> None:
        self._shuffle = bool(shuffle)
        check_std(noise_std, "noise_std")
        self._noise_std = float(noise_std)

        check_interval(ratio, low=0.0, high=1.0, name="ratio")
        self._ratio = float(ratio)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(shuffle={self._shuffle}, "
            f"noise_std={self._noise_std}, ratio={self._ratio})"
        )

    @property
    def noise_std(self) -> float:
        r"""``float``: The standard deviation of the Gaussian noise."""
        return self._noise_std

    @property
    def ratio(self) -> float:
        r"""``float``: the ratio between the number of examples in outer
        circle and inner circle."""
        return self._ratio

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> BatchDict[BatchedTensor]:
        return make_moons_classification(
            num_examples=batch_size,
            shuffle=self._shuffle,
            noise_std=self._noise_std,
            ratio=self._ratio,
            generator=rng,
        )


def make_moons_classification(
    num_examples: int = 100,
    shuffle: bool = True,
    noise_std: float = 0.0,
    ratio: float = 0.5,
    generator: torch.Generator | None = None,
) -> BatchDict[BatchedTensor]:
    r"""Generates a binary classification dataset where the data are two
    interleaving half circles in 2d.

    The implementation is based on
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

    Args:
    ----
        num_examples (int, optional): Specifies the number of examples.
            Default: ``100``
        shuffle (bool, optional): If ``True``, the examples are
            shuffled. Default: ``True``
        noise_std (float, optional): Specifies the standard deviation
            of the Gaussian noise.
        ratio (float, optional): Specifies the ratio between the
            number of examples in outer circle and inner circle.
            Default: ``0.5``
        generator (``torch.Generator`` or ``None``, optional):
            Specifies an optional random generator.

    Returns:
    -------
        ``BatchDict``: A batch with two items:
            - ``'input'``: a ``BatchedTensor`` of type float and
                shape ``(num_examples, 2)``. This
                tensor represents the input features.
            - ``'target'``: a ``BatchedTensor`` of type long and
                shape ``(num_examples,)``. This tensor represents
                the targets.

    Raises:
    ------
        RuntimeError if one of the parameters is not valid.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import make_moons_classification
        >>> batch = make_moons_classification(num_examples=10)
        >>> batch
        BatchDict(
          (target): tensor([...], batch_dim=0)
          (feature): tensor([[...]], batch_dim=0)
        )
    """
    check_num_examples(num_examples)
    check_std(noise_std, "noise_std")
    check_interval(ratio, low=0.0, high=1.0, name="ratio")

    num_examples_out = math.ceil(num_examples * ratio)
    num_examples_in = num_examples - num_examples_out

    linspace_out = torch.linspace(0, math.pi, num_examples_out)
    linspace_in = torch.linspace(0, math.pi, num_examples_in)
    outer_circ = torch.stack([linspace_out.cos(), linspace_out.sin()], dim=1)
    inner_circ = torch.stack([1.0 - linspace_in.cos(), 0.5 - linspace_in.sin()], dim=1)

    features = torch.cat([outer_circ, inner_circ], dim=0)
    targets = torch.cat(
        [
            torch.zeros(num_examples_out, dtype=torch.long),
            torch.ones(num_examples_in, dtype=torch.long),
        ],
        dim=0,
    )

    if noise_std > 0.0:
        features += rand_normal(size=(num_examples, 2), std=noise_std, generator=generator)

    batch = BatchDict({ct.TARGET: BatchedTensor(targets), ct.FEATURE: BatchedTensor(features)})
    if shuffle:
        batch.shuffle_along_batch_(generator)
    return batch
