from __future__ import annotations

__all__ = ["TimeSeriesGenerator"]


from coola.utils import str_indent, str_mapping
from redcat import BatchDict
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.timeseries.base import BaseTimeSeriesGenerator


class TimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implements a generic time series generator.

    Args:
    ----
        sequences (dict): Specifies the sequence generators or their
            configurations.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import RandUniform
        >>> from startorch.timeseries import TimeSeries
        >>> generator = TimeSeries({"value": RandUniform(), "time": RandUniform()})
        >>> generator
        TimeSeriesGenerator(
          (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
          (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        BatchDict(
          (value): tensor([[...]], batch_dim=0, seq_dim=1)
          (time): tensor([[...]], batch_dim=0, seq_dim=1)
        )
    """

    def __init__(self, sequences: dict[str, BaseSequenceGenerator | dict]) -> None:
        super().__init__()
        self._sequences = {
            key: setup_sequence_generator(generator) for key, generator in sequences.items()
        }

    def __repr__(self) -> str:
        args = str_indent(str_mapping(self._sequences))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchDict:
        return BatchDict(
            {
                key: generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
                for key, generator in self._sequences.items()
            }
        )
