from __future__ import annotations

__all__ = ["BaseSequenceGenerator", "setup_sequence_generator"]

import logging
from abc import ABC, abstractmethod

from objectory import AbstractFactory
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseSequenceGenerator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to generate sequences.

    A child class has to implement the ``generate`` method.
    """

    @abstractmethod
    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        r"""Generates a batch of sequences.

        All the sequences in the batch must have the same length.

        Args:
        ----
            seq_len (int): Specifies the sequence length.
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
            >>> from startorch.sequence import RandUniform
            >>> generator = RandUniform()
            >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
            tensor([[...]], batch_dim=0, seq_dim=1)
        """


def setup_sequence_generator(generator: BaseSequenceGenerator | dict) -> BaseSequenceGenerator:
    r"""Sets up a sequence generator.

    The sequence generator is instantiated from its configuration by
    using the ``BaseSequenceGenerator`` factory function.

    Args:
    ----
        generator (``BaseSequenceGenerator`` or dict): Specifies a
            sequence generator or its configuration.

    Returns:
        ``BaseSequenceGenerator``: A sequence generator.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import setup_sequence_generator
        >>> setup_sequence_generator({"_target_": "startorch.sequence.RandUniform"})
        RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing a sequence generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BaseSequenceGenerator.factory(**generator)
    return generator