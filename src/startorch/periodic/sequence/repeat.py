r"""Contain a periodic sequence generator that generates periodic
sequences by using a ``BaseSequenceGenerator`` object and repeating the
generated sequences."""

from __future__ import annotations

__all__ = ["RepeatPeriodicSequenceGenerator"]

import math
from typing import TYPE_CHECKING

from batchtensor.tensor import repeat_along_seq, slice_along_seq
from coola.utils import str_indent, str_mapping

from startorch.periodic.sequence import BasePeriodicSequenceGenerator
from startorch.sequence import BaseSequenceGenerator, setup_sequence_generator

if TYPE_CHECKING:
    import torch


class RepeatPeriodicSequenceGenerator(BasePeriodicSequenceGenerator):
    r"""Implement a class to generate periodic sequences by using a
    ``BaseSequenceGenerator`` object and repeating the generated
    sequence.

    Args:
        generator: A sequence generator or its configuration.

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

    def __init__(self, generator: BaseSequenceGenerator | dict) -> None:
        super().__init__()
        self._generator = setup_sequence_generator(generator)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"generator": self._generator}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, period: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        seq = self._generator.generate(seq_len=period, batch_size=batch_size, rng=rng)
        seq = repeat_along_seq(seq, math.ceil(seq_len / period))
        return slice_along_seq(seq, stop=seq_len)
