r"""Contain transition matrix generators."""

from __future__ import annotations

__all__ = [
    "BaseTransitionGenerator",
    "is_transition_generator_config",
    "setup_transition_generator",
]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

from startorch.utils.format import str_target_object

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class BaseTransitionGenerator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to generate a transition matrix.

    A child class has to implement the ``generate`` method.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transition import Diagonal
    >>> generator = Diagonal()
    >>> generator
    DiagonalTransitionGenerator()
    >>> generator.generate(n=6)
    tensor([[1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.]])

    ```
    """

    @abstractmethod
    def generate(self, n: int, rng: torch.Generator | None = None) -> torch.Tensor:
        r"""Return a transition matrix.

        Args:
            n: The size of the transition matrix.
            rng: An optional random number generator.

        Returns:
            The generated transition matrix of shape ``(n, n)`` and
                data type float.

        Example usage:

        ```pycon

        >>> import torch
        >>> from startorch.transition import Diagonal
        >>> generator = Diagonal()
        >>> generator.generate(n=6)
        tensor([[1., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 1.]])

        ```
        """


def is_transition_generator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseTransitionGenerator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseTransitionGenerator`` object.

    Example usage:

    ```pycon

    >>> from startorch.transition import is_transition_generator_config
    >>> is_transition_generator_config({"_target_": "startorch.transition.Diagonal"})
    True

    ```
    """
    return is_object_config(config, BaseTransitionGenerator)


def setup_transition_generator(
    generator: BaseTransitionGenerator | dict,
) -> BaseTransitionGenerator:
    r"""Set up a transition matrix generator.

    The transition generator is instantiated from its configuration by
    using the ``BaseTransitionGenerator`` factory function.

    Args:
        generator: A transition matrix generator or its configuration.

    Returns:
        A transition matrix generator.

    Example usage:

    ```pycon

    >>> from startorch.transition import setup_transition_generator
    >>> setup_transition_generator({"_target_": "startorch.transition.Diagonal"})
    DiagonalTransitionGenerator()

    ```
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing a transition matrix generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BaseTransitionGenerator.factory(**generator)
    if not isinstance(generator, BaseTransitionGenerator):
        logger.warning(
            f"generator is not a `BaseTransitionGenerator` (received: {type(generator)})"
        )
    return generator
