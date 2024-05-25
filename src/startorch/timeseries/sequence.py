r"""Contain the implementation of a time series generator that generates
time series with sequence generators."""

from __future__ import annotations

__all__ = ["SequenceTimeSeriesGenerator"]


from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.timeseries.base import BaseTimeSeriesGenerator

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import torch


class SequenceTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implement a time series generator that generates time series with
    sequence generators.

    Args:
        generators: The sequence generators or their
            configurations.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import RandUniform
    >>> from startorch.timeseries import SequenceTimeSeries
    >>> generator = SequenceTimeSeries({"value": RandUniform(), "time": RandUniform()})
    >>> generator
    SequenceTimeSeriesGenerator(
      (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
      (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    {'value': tensor([[...]]), 'time': tensor([[...]])}

    ```
    """

    def __init__(self, generators: Mapping[str, BaseSequenceGenerator | dict]) -> None:
        super().__init__()
        self._generators = {
            key: setup_sequence_generator(generator) for key, generator in generators.items()
        }

    def __repr__(self) -> str:
        args = str_indent(str_mapping(self._generators))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        return {
            key: generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
            for key, generator in self._generators.items()
        }
