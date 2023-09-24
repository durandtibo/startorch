from __future__ import annotations

__all__ = ["RepeatPeriodicSequenceGenerator"]

import math

from coola.utils import str_indent, str_mapping
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.periodic.sequence import BasePeriodicSequenceGenerator
from startorch.sequence import BaseSequenceGenerator, setup_sequence_generator


class RepeatPeriodicSequenceGenerator(BasePeriodicSequenceGenerator):
    r"""Implements a class to generate periodic sequences by using a
    ``BaseSequenceGenerator`` object and repeating the generated
    sequence.

    Args:
    ----
        sequence (``BaseSequenceGenerator`` or dict): Specifies
            a sequence generator or its configuration.

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

    def __init__(self, generator: BaseSequenceGenerator | dict) -> None:
        super().__init__()
        self._generator = setup_sequence_generator(generator)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"generator": self._generator}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, period: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return (
            self._generator.generate(seq_len=period, batch_size=batch_size, rng=rng)
            .repeat_along_seq(math.ceil(seq_len / period))
            .slice_along_seq(stop=seq_len)
        )
