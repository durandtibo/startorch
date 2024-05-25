r"""Contain the base class to implement a time series generator."""

from __future__ import annotations

__all__ = ["MergeTimeSeriesGenerator"]

from typing import TYPE_CHECKING

from batchtensor.nested import slice_along_seq
from coola.utils import str_indent, str_sequence

from startorch import constants as ct
from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)
from startorch.timeseries.utils import merge_by_time

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import torch


class MergeTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implement a time series creator that creates time series by
    combining several time series.

    The time series are combined by using the time information.

    Args:
        generators: The time series generators or their
            configuration.
        time_key: The key used to merge the time series by
            time.

    Example usage:

    ```pycon

    >>> from startorch.timeseries import Merge, SequenceTimeSeriesGenerator
    >>> from startorch.sequence import RandUniform, RandNormal
    >>> generator = Merge(
    ...     (
    ...         SequenceTimeSeriesGenerator({"value": RandUniform(), "time": RandUniform()}),
    ...         SequenceTimeSeriesGenerator({"value": RandNormal(), "time": RandNormal()}),
    ...     )
    ... )
    >>> generator
    MergeTimeSeriesGenerator(
      (time_key): time
      (0): SequenceTimeSeriesGenerator(
          (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
          (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
      (1): SequenceTimeSeriesGenerator(
          (value): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
          (time): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
        )
    )
    >>> batch = generator.generate(seq_len=12, batch_size=10)
    >>> batch
    {'value': tensor([[...]]), 'time': tensor([[...]])}

    ```
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
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        timeseries = [
            generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
            for generator in self._generators
        ]
        data = merge_by_time(timeseries, time_key=self._time_key)
        return slice_along_seq(data, stop=seq_len)
