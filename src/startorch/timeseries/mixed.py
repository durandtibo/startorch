from __future__ import annotations

__all__ = ["MixedTimeSeriesGenerator"]


from coola.utils import str_indent, str_mapping
from redcat import BatchDict
from torch import Generator

from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)
from startorch.timeseries.utils import mix2sequences


class MixedTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implements a generic time series generator.

    Args:
    ----
        generator (``BaseTimeSeriesGenerator`` or dict): Specifies
            the time series generator or its configuration.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import RandUniform
        >>> from startorch.timeseries import MixedTimeSeries, TimeSeries
        >>> generator = MixedTimeSeries(
        ...     TimeSeries({"key1": RandUniform(), "key2": RandUniform()}),
        ...     key1="key1",
        ...     key2="key2",
        ... )
        >>> generator
        MixedTimeSeriesGenerator(
          (generator): TimeSeriesGenerator(
              (key1): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
              (key2): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
            )
          (key1): key1
          (key2): key2
        )
        >>> generator.generate(seq_len=12, batch_size=10)
        BatchDict(
          (key1): tensor([[...]], batch_dim=0, seq_dim=1)
          (key2): tensor([[...]], batch_dim=0, seq_dim=1)
        )
    """

    def __init__(
        self,
        generator: BaseTimeSeriesGenerator | dict,
        key1: str,
        key2: str,
    ) -> None:
        super().__init__()
        self._generator = setup_timeseries_generator(generator)
        self._key1 = key1
        self._key2 = key2

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping({"generator": self._generator, "key1": self._key1, "key2": self._key2})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchDict:
        batch = self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        seq1, seq2 = mix2sequences(batch[self._key1], batch[self._key2])
        batch[self._key1] = seq1
        batch[self._key2] = seq2
        return batch
