from __future__ import annotations

__all__ = ["MergeTimeSeriesGenerator"]

from collections.abc import Generator, Sequence

from coola.utils import str_indent, str_sequence
from redcat import BatchDict

from startorch import constants as ct
from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)
from startorch.timeseries.utils import merge_timeseries_by_time


class MergeTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implements a time series creator that creates time series by
    combining several time series.

    The time series are combined by using the time information.

    Args:
    ----
        generators (``Sequence``): Specifies the time series
            generators or their configuration.
        time_key (str, optional): Specifies the key used to merge the
            time series by time. Default: ``'time'``

    Example usage:

    .. code-block:: pycon

        >>> from startorch.timeseries import Merge, TimeSeries
        >>> from startorch.sequence import RandUniform, RandNormal
        >>> generator = Merge(
        ...     (
        ...         TimeSeries({"value": RandUniform(), "time": RandUniform()}),
        ...         TimeSeries({"value": RandNormal(), "time": RandNormal()}),
        ...     )
        ... )
        >>> generator
        MergeTimeSeriesGenerator(
          (time_key): time
          (0): TimeSeriesGenerator(
              (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
              (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
            )
          (1): TimeSeriesGenerator(
              (value): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
              (time): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
            )
        )
        >>> batch = generator.generate(seq_len=12, batch_size=10)
        >>> batch
        BatchDict(
          (value): tensor([[...]], batch_dim=0, seq_dim=1)
          (time): tensor([[...]], batch_dim=0, seq_dim=1)
        )
    """

    def __init__(
        self, generators: Sequence[BaseTimeSeriesGenerator | dict], time_key: str = ct.TIME
    ) -> None:
        super().__init__()
        self._generators = tuple(setup_timeseries_generator(generator) for generator in generators)
        self._time_key = time_key

    def __repr__(self) -> str:
        args = str_indent(str_sequence(self._generators))
        return f"{self.__class__.__qualname__}(\n  (time_key): {self._time_key}\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchDict:
        timeseries = [
            generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
            for generator in self._generators
        ]
        return merge_timeseries_by_time(timeseries, time_key=self._time_key).slice_along_seq(
            stop=seq_len
        )
