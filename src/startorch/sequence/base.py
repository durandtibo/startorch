r"""Contain the base class to implement a sequence generator."""

from __future__ import annotations

__all__ = ["BaseSequenceGenerator", "is_sequence_generator_config", "setup_sequence_generator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

from startorch.utils.format import str_target_object

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class BaseSequenceGenerator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to generate sequences.

    A child class has to implement the ``generate`` method.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.sequence import RandUniform
    >>> generator = RandUniform()
    >>> generator
    RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    @abstractmethod
    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Generate a batch of sequences.

        All the sequences in the batch must have the same length.

        Args:
            seq_len: Specifies the sequence length.
            batch_size: Specifies the batch size.
            rng: Specifies an optional random number generator.

        Returns:
            A batch of sequences. The data in the batch are
                represented as a ``torch.Tensor`` of shape
                ``(batch_size, sequence_length, *)`` where `*` means
                any number of dimensions.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.sequence import RandUniform
        >>> generator = RandUniform()
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]])

        ```
        """


def is_sequence_generator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseSequenceGenerator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: Specifies the configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseSequenceGenerator`` object.

    Example usage:

    ```pycon
    >>> from startorch.sequence import is_sequence_generator_config
    >>> is_sequence_generator_config({"_target_": "startorch.sequence.RandUniform"})
    True

    ```
    """
    return is_object_config(config, BaseSequenceGenerator)


def setup_sequence_generator(generator: BaseSequenceGenerator | dict) -> BaseSequenceGenerator:
    r"""Set up a sequence generator.

    The sequence generator is instantiated from its configuration by
    using the ``BaseSequenceGenerator`` factory function.

    Args:
        generator: Specifies a sequence generator or its configuration.

    Returns:
        A sequence generator.

    Example usage:

    ```pycon
    >>> from startorch.sequence import setup_sequence_generator
    >>> setup_sequence_generator({"_target_": "startorch.sequence.RandUniform"})
    RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))

    ```
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing a sequence generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BaseSequenceGenerator.factory(**generator)
    if not isinstance(generator, BaseSequenceGenerator):
        logger.warning(f"generator is not a `BaseSequenceGenerator` (received: {type(generator)})")
    return generator
