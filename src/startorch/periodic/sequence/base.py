r"""Contain the base class to implement a periodic sequence
generator."""

from __future__ import annotations

__all__ = [
    "BasePeriodicSequenceGenerator",
    "is_periodic_sequence_generator_config",
    "setup_periodic_sequence_generator",
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


class BasePeriodicSequenceGenerator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to generate periodic sequences.

    A child class has to implement the ``generate`` method.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.periodic.sequence import Repeat
    >>> from startorch.sequence import RandUniform
    >>> generator = Repeat(RandUniform())
    >>> generator
    RepeatPeriodicSequenceGenerator(
      (generator): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, period=4, batch_size=4)
    tensor([[...]])

    ```
    """

    @abstractmethod
    def generate(
        self, seq_len: int, period: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Generate a batch of periodic sequences.

        All the sequences in the batch have the same length.

        Args:
            seq_len: Specifies the sequence length.
            period: Specifies the period.
            batch_size: Specifies the batch size.
            rng: Specifies an optional random number generator.

        Returns:
            A batch of sequences represented as a tensor of shape
                ``(batch_size, sequence_length, *)`` where `*` means
                any number of dimensions.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.periodic.sequence import Repeat
        >>> from startorch.sequence import RandUniform
        >>> generator = Repeat(RandUniform())
        >>> generator.generate(seq_len=12, period=4, batch_size=4)
        tensor([[...]])

        ```
        """


def is_periodic_sequence_generator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BasePeriodicSequenceGenerator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: Specifies the configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BasePeriodicSequenceGenerator`` object.

    Example usage:

    ```pycon
    >>> from startorch.periodic.sequence import is_periodic_sequence_generator_config
    >>> is_periodic_sequence_generator_config(
    ...     {
    ...         "_target_": "startorch.periodic.sequence.Repeat",
    ...         "generator": {"_target_": "startorch.sequence.RandUniform"},
    ...     }
    ... )
    True

    ```
    """
    return is_object_config(config, BasePeriodicSequenceGenerator)


def setup_periodic_sequence_generator(
    generator: BasePeriodicSequenceGenerator | dict,
) -> BasePeriodicSequenceGenerator:
    r"""Set up a periodic sequence generator.

    The sequence generator is instantiated from its configuration by
    using the ``BasePeriodicSequenceGenerator`` factory function.

    Args:
        generator: Specifies a periodic sequence generator or its
            configuration.

    Returns:
        A periodic sequence generator.

    Example usage:

    ```pycon
    >>> from startorch.periodic.sequence import setup_periodic_sequence_generator
    >>> setup_periodic_sequence_generator(
    ...     {
    ...         "_target_": "startorch.periodic.sequence.Repeat",
    ...         "generator": {"_target_": "startorch.sequence.RandUniform"},
    ...     }
    ... )
    RepeatPeriodicSequenceGenerator(
      (generator): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )

    ```
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing a periodic sequence generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BasePeriodicSequenceGenerator.factory(**generator)
    if not isinstance(generator, BasePeriodicSequenceGenerator):
        logger.warning(
            f"generator is not a `BasePeriodicSequenceGenerator` (received: {type(generator)})"
        )
    return generator
