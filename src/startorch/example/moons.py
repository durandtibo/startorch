from __future__ import annotations

__all__ = ["make_moons_classification"]

import math

import torch
from redcat import BatchDict, BatchedTensor

from startorch import constants as ct
from startorch.example.utils import check_interval, check_num_examples, check_std
from startorch.random import rand_normal


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
            of the Gaussian noise. Default: ``0.0``
        ratio (float, optional): Specifies the ratio between the
            number of examples in outer circle and inner circle.
            Default: ``0.5``
        generator (``torch.Generator`` or ``None``, optional):
            Specifies an optional random generator. Default: ``None``

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
    inner_circ = torch.stack([linspace_in.cos().sub(1.0), linspace_in.sin().add(0.5)], dim=1)

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