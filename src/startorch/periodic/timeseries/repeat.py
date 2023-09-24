from __future__ import annotations

__all__ = ["RepeatPeriodicTimeSeriesGenerator"]

import math

from coola.utils import str_indent, str_mapping
from redcat import BatchDict
from torch import Generator

from startorch.periodic.timeseries import BasePeriodicTimeSeriesGenerator
from startorch.timeseries import BaseTimeSeriesGenerator, setup_timeseries_generator


class RepeatPeriodicTimeSeriesGenerator(BasePeriodicTimeSeriesGenerator):
    r"""Implements a class to generate periodic sequences by using a
    ``BaseTimeSeriesGenerator`` object and repeating the generated
    sequence.

    Args:
    ----
        sequence (``BaseTimeSeriesGenerator`` or dict): Specifies
            a sequence generator or its configuration.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.periodic.timeseries import Repeat
        >>> from startorch.timeseries import TimeSeries
        >>> from startorch.sequence import RandUniform
        >>> generator = Repeat(TimeSeries({"value": RandUniform(), "time": RandUniform()}))
        >>> generator
        RepeatPeriodicTimeSeriesGenerator(
          (generator): TimeSeriesGenerator(
              (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
              (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
            )
        )
        >>> generator.generate(seq_len=12, period=4, batch_size=4)
        BatchDict(
          (value): tensor([[...]], batch_dim=0, seq_dim=1)
          (time): tensor([[...]], batch_dim=0, seq_dim=1)
        )
    """

    def __init__(self, generator: BaseTimeSeriesGenerator | dict) -> None:
        super().__init__()
        self._generator = setup_timeseries_generator(generator)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"generator": self._generator}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, period: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchDict:
        return (
            self._generator.generate(seq_len=period, batch_size=batch_size, rng=rng)
            .repeat_along_seq(math.ceil(seq_len / period))
            .slice_along_seq(stop=seq_len)
        )
