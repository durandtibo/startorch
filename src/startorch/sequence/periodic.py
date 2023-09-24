from __future__ import annotations

__all__ = ["PeriodicSequenceGenerator"]

import math

from coola.utils import str_indent, str_mapping
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.periodic.sequence.base import BasePeriodicSequenceGenerator
from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator


class PeriodicSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator to generate periodic sequence
    from a regular sequence generator.

    Args:
    ----
        sequence (``BaseSequenceGenerator`` or
            ``BasePeriodicSequenceGenerator`` or dict): Specifies a
            sequence generator or its configuration that is used to
            generate the periodic pattern.
        period (``BaseTensorGenerator`` or dict): Specifies
            the period length sampler or its configuration. This
            sampler is used to sample the period length at each
            batch.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import Periodic, RandUniform
        >>> from startorch.tensor import RandInt
        >>> generator = Periodic(sequence=RandUniform(), period=RandInt(2, 5))
        >>> generator
        PeriodicSequenceGenerator(
          (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
          (period): RandIntTensorGenerator(low=2, high=5)
        )
        >>> generator.generate(seq_len=10, batch_size=2)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        sequence: BaseSequenceGenerator | BasePeriodicSequenceGenerator | dict,
        period: BaseTensorGenerator | dict,
    ) -> None:
        super().__init__()
        self._sequence = setup_sequence_generator(sequence)
        self._period = setup_tensor_generator(period)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"sequence": self._sequence, "period": self._period}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        period = int(self._period.generate((1,), rng=rng).item())
        if isinstance(self._sequence, BasePeriodicSequenceGenerator):
            return self._sequence.generate(
                seq_len=seq_len, period=period, batch_size=batch_size, rng=rng
            )
        return (
            self._sequence.generate(seq_len=period, batch_size=batch_size, rng=rng)
            .repeat_along_seq(math.ceil(seq_len / period))
            .slice_along_seq(stop=seq_len)
        )
