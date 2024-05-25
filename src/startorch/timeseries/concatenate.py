r"""Contain the implementation of time series generator that
concatenates the outputs of multiple time series generators."""

from __future__ import annotations

__all__ = ["ConcatenateTimeSeriesGenerator"]

from typing import TYPE_CHECKING

from coola.utils import str_indent, str_sequence

from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import torch


class ConcatenateTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implement a time series generator that concatenates the outputs
    of multiple time series generators.

    Note that the last value is used if there are duplicated keys.

    Args:
        generators: The time series generators or their configurations.

    TimeSeries usage:

    ```pycon

    >>> from startorch.timeseries import TimeSeriesGenerator, Concatenate
    >>> from startorch.sequence import RandInt, RandUniform
    >>> generator = Concatenate(
    ...     [
    ...         TimeSeriesGenerator(
    ...             generators={"value": RandUniform(), "time": RandUniform()},
    ...         ),
    ...         TimeSeriesGenerator(generators={"label": RandInt(0, 10)}),
    ...     ]
    ... )
    >>> generator
    ConcatenateTimeSeriesGenerator(
      (0): TimeSeriesGenerator(
          (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
          (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
      (1): TimeSeriesGenerator(
          (label): RandIntSequenceGenerator(low=0, high=10, feature_size=())
        )
    )
    >>> generator.generate(seq_len=10, batch_size=5)
    {'value': tensor([[...]]), 'time': tensor([[...]]), 'label': tensor([[...]])}

    ```
    """

    def __init__(
        self,
        generators: Sequence[BaseTimeSeriesGenerator | dict],
    ) -> None:
        super().__init__()
        self._generators = [setup_timeseries_generator(generator) for generator in generators]

    def __repr__(self) -> str:
        args = str_indent(str_sequence(self._generators))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        out = {}
        for generator in self._generators:
            out |= generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        return out
