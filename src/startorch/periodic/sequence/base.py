from __future__ import annotations

__all__ = [
    "BasePeriodicSequenceGenerator",
    "is_periodic_sequence_generator_config",
    "setup_periodic_sequence_generator",
]

import logging
from abc import ABC, abstractmethod

from objectory import AbstractFactory
from objectory.utils import is_object_config
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BasePeriodicSequenceGenerator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to generate periodic sequences.

    A child class has to implement the ``generate`` method.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.periodic.sequence import Repeat
        >>> from startorch.sequence import RandUniform
        >>> generator = Repeat(RandUniform())
        >>> generator
        RepeatPeriodicSequenceGenerator(
          (generator): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
        >>> generator.generate(seq_len=12, period=4, batch_size=4)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    @abstractmethod
    def generate(
        self, seq_len: int, period: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        r"""Generates a batch of periodic sequences.

        All the sequences in the batch have the same length.

        Args:
        ----
            seq_len (int): Specifies the sequence length.
            period (int): Specifies the period.
            batch_size (int, optional): Specifies the batch size.
                Default: ``1``
            rng (``torch.Generator`` or None, optional): Specifies
                an optional random number generator. Default: ``None``

        Returns:
        -------
            ``BatchedTensorSeq``: A batch of sequences. The data in the
                batch are represented by a ``torch.Tensor`` of shape
                ``(batch_size, sequence_length, *)`` where `*` means
                any number of dimensions.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from startorch.periodic.sequence import Repeat
            >>> from startorch.sequence import RandUniform
            >>> generator = Repeat(RandUniform())
            >>> generator.generate(seq_len=12, period=4, batch_size=4)
            tensor([[...]], batch_dim=0, seq_dim=1)
        """


def is_periodic_sequence_generator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BasePeriodicSequenceGenerator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BasePeriodicSequenceGenerator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.periodic.sequence import is_periodic_sequence_generator_config
        >>> is_periodic_sequence_generator_config(
        ...     {
        ...         "_target_": "startorch.periodic.sequence.Repeat",
        ...         "generator": {"_target_": "startorch.sequence.RandUniform"},
        ...     }
        ... )
        True
    """
    return is_object_config(config, BasePeriodicSequenceGenerator)


def setup_periodic_sequence_generator(
    generator: BasePeriodicSequenceGenerator | dict,
) -> BasePeriodicSequenceGenerator:
    r"""Sets up a periodic sequence generator.

    The sequence generator is instantiated from its configuration by
    using the ``BasePeriodicSequenceGenerator`` factory function.

    Args:
    ----
        generator (``BasePeriodicSequenceGenerator`` or dict): Specifies a
            periodic sequence generator or its configuration.

    Returns:
    -------
        ``BasePeriodicSequenceGenerator``: A periodic sequence generator.

    Example usage:

    .. code-block:: pycon

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
