r"""Contain the base class to implement an example generator."""

from __future__ import annotations

__all__ = [
    "BaseExampleGenerator",
    "is_example_generator_config",
    "setup_example_generator",
]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from objectory.utils import is_object_config

from startorch.utils.format import str_target_object

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseExampleGenerator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to generate examples.

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

    @abstractmethod
    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        r"""Generate a batch of examples.

        Args:
            batch_size: Specifies the batch size.
            rng: Specifies an optional random number generator.

        Returns:
            A batch of examples.

        Example usage:

        ```pycon
        >>> from startorch.example import HypercubeClassification
        >>> generator = HypercubeClassification(num_classes=5, feature_size=6)
        >>> batch = generator.generate(batch_size=10)
        >>> batch
        {'target': tensor([...]), 'feature': tensor([[...]])}

        ```
        """


def is_example_generator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseExampleGenerator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: Specifies the configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseExampleGenerator`` object.

    Example usage:

    ```pycon

        >>> from startorch.example import is_example_generator_config
        >>> is_example_generator_config({"_target_": "startorch.example.HypercubeClassification"})
        True
    """
    return is_object_config(config, BaseExampleGenerator)


def setup_example_generator(
    generator: BaseExampleGenerator | dict,
) -> BaseExampleGenerator:
    r"""Set up a time series generator.

    The time series generator is instantiated from its configuration
    by using the ``BaseExampleGenerator`` factory function.

    Args:
        generator: Specifies a time series generator or its
            configuration.

    Returns:
        A time series generator.

    Example usage:

    ```pycon
    >>> from startorch.example import setup_example_generator
    >>> generator = setup_example_generator(
    ...     {"_target_": "startorch.example.HypercubeClassification"}
    ... )
    >>> generator
    HypercubeClassificationExampleGenerator(num_classes=50, feature_size=64, noise_std=0.2)

    ```
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing an example generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BaseExampleGenerator.factory(**generator)
    if not isinstance(generator, BaseExampleGenerator):
        logger.warning(f"generator is not a `BaseExampleGenerator` (received: {type(generator)})")
    return generator
