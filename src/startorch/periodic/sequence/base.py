from __future__ import annotations

__all__ = ["BasePeriodicSequenceGenerator", "setup_periodic_sequence_generator"]

import logging
from abc import ABC, abstractmethod

from objectory import AbstractFactory
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BasePeriodicSequenceGenerator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to generate periodic sequences.

    A child class has to implement the ``generate`` method.
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
        """


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
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing a periodic sequence generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BasePeriodicSequenceGenerator.factory(**generator)
    return generator
