r"""Contain an example generator to generate multiclass classification
data using the hypercube pattern."""

from __future__ import annotations

__all__ = ["HypercubeClassificationExampleGenerator", "make_hypercube_classification"]


import torch

from startorch import constants as ct
from startorch.example.base import BaseExampleGenerator
from startorch.utils.validation import (
    check_feature_size,
    check_integer_ge,
    check_num_examples,
    check_std,
)


class HypercubeClassificationExampleGenerator(BaseExampleGenerator):
    r"""Implement a classification example generator.

    The data are generated by using a hypercube. The targets are some
    vertices of the hypercube. Each input feature is a 1-hot
    representation of the target plus a Gaussian noise. These data can
    be used for a multi-class classification task.

    Args:
        num_classes: The number of classes.
        feature_size: The feature size. The feature size has
            to be greater than the number of classes.
        noise_std: The standard deviation of the Gaussian
            noise.

    Raises:
        ValueError: if one of the parameters is not valid.

    Example usage:

    ```pycon

    >>> from startorch.example import HypercubeClassification
    >>> generator = HypercubeClassification(num_classes=5, feature_size=6)
    >>> generator
    HypercubeClassificationExampleGenerator(num_classes=5, feature_size=6, noise_std=0.2)
    >>> batch = generator.generate(batch_size=10)
    >>> batch
    {'target': tensor([...]), 'feature': tensor([[...]])}

    ```
    """

    def __init__(
        self,
        num_classes: int = 50,
        feature_size: int = 64,
        noise_std: float = 0.2,
    ) -> None:
        check_integer_ge(num_classes, low=1, name="num_classes")
        self._num_classes = int(num_classes)

        check_feature_size(feature_size, low=self._num_classes)
        self._feature_size = int(feature_size)

        check_std(noise_std, "noise_std")
        self._noise_std = float(noise_std)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"num_classes={self._num_classes:,}, "
            f"feature_size={self._feature_size:,}, "
            f"noise_std={self._noise_std:,})"
        )

    @property
    def num_classes(self) -> int:
        r"""The number of classes when the data are created."""
        return self._num_classes

    @property
    def feature_size(self) -> int:
        r"""The feature size when the data are created."""
        return self._feature_size

    @property
    def noise_std(self) -> float:
        r"""The standard deviation of the Gaussian noise."""
        return self._noise_std

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        return make_hypercube_classification(
            num_examples=batch_size,
            num_classes=self._num_classes,
            feature_size=self._feature_size,
            noise_std=self._noise_std,
            generator=rng,
        )


def make_hypercube_classification(
    num_examples: int = 1000,
    num_classes: int = 50,
    feature_size: int = 64,
    noise_std: float = 0.2,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    r"""Generate a synthetic classification dataset based on hypercube
    vertex structure.

    The data are generated by using a hypercube. The targets are some
    vertices of the hypercube. Each input feature is a 1-hot
    representation of the target plus a Gaussian noise. These data can
    be used for a multi-class classification task.

    Args:
        num_examples: The number of examples.
        num_classes: The number of classes.
        feature_size: The feature size. The feature size has
            to be greater than the number of classes.
        noise_std: The standard deviation of the Gaussian
            noise.
        generator: An optional random generator.

    Returns:
        A dictionary with two items:
            - ``'input'``: a ``torch.Tensor`` of type float and
                shape ``(num_examples, feature_size)``. This
                tensor represents the input features.
            - ``'target'``: a ``torch.Tensor`` of type long and
                shape ``(num_examples,)``. This tensor represents
                the targets.

    Raises:
        RuntimeError: if one of the parameters is not valid.

    Example usage:

    ```pycon

    >>> from startorch.example.hypercube import make_hypercube_classification
    >>> batch = make_hypercube_classification(num_examples=10, num_classes=5, feature_size=10)
    >>> batch
    {'target': tensor([...]), 'feature': tensor([[...]])}

    ```
    """
    check_num_examples(num_examples)
    check_integer_ge(num_classes, low=1, name="num_classes")
    check_feature_size(feature_size, low=num_classes)
    check_std(noise_std, "noise_std")
    # Generate the target of each example.
    targets = torch.randint(0, num_classes, (num_examples,), generator=generator)
    # Generate the features. Each class should be a vertex of the hyper-cube
    # plus Gaussian noise.
    features = torch.randn(num_examples, feature_size, generator=generator).mul(noise_std)
    features.scatter_add_(1, targets.view(num_examples, 1), torch.ones(num_examples, 1))
    return {ct.TARGET: targets, ct.FEATURE: features}
