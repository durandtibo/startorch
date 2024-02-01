r"""Implement a time series generator that generates time series by
mixing two sequences of a time series."""

from __future__ import annotations

__all__ = ["MixedTimeSeriesGenerator"]


from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)
from startorch.timeseries.utils import mix2sequences

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch


class MixedTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implement a time series generator that generates time series by
    mixing two sequences of a time series.

    Args:
        generator: Specifies the time series generator or its
            configuration.
        key1: Specifies the key of the first sequence to mix.
        key2: Specifies the key of the second sequence to mix.

    Example usage:

    ```pycon
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
    {'key1': tensor([[...]]), 'key2': tensor([[...]])}

    ```
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
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        batch = self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        seq1, seq2 = mix2sequences(batch[self._key1], batch[self._key2])
        batch[self._key1] = seq1
        batch[self._key2] = seq2
        return batch
