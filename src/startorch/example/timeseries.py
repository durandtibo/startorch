from __future__ import annotations

__all__ = ["TimeSeriesExampleGenerator"]


from coola.utils import str_indent, str_mapping
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.example.base import BaseExampleGenerator
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator
from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)


class TimeSeriesExampleGenerator(BaseExampleGenerator):
    r"""Implements an example generator to generate time series.

    Args:
    ----
        timeseries (``BaseTimeSeriesGenerator`` or dict): Specifies a
            time series generator or its configuration.
        seq_len (``BaseTensorGenerator`` or dict): Specifies
            the sequence length sampler or its configuration. This
            sampler is used to sample the sequence length at each
            batch.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import TimeSeriesExampleGenerator
        >>> from startorch.timeseries import TimeSeriesGenerator
        >>> from startorch.sequence import Periodic, RandUniform
        >>> from startorch.tensor import RandInt
        >>> generator = TimeSeriesExampleGenerator(
        ...     timeseries=TimeSeriesGenerator({"value": RandUniform(), "time": RandUniform()}),
        ...     seq_len=RandInt(2, 5),
        ... )
        >>> generator
        TimeSeriesExampleGenerator(
          (timeseries): TimeSeriesGenerator(
              (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
              (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
            )
          (seq_len): RandIntTensorGenerator(low=2, high=5)
        )
        >>> generator.generate(batch_size=10)
        BatchDict(
          (value): tensor([...], batch_dim=0, seq_dim=1)
          (time): tensor([[...]], batch_dim=0, seq_dim=1)
        )
    """

    def __init__(
        self,
        timeseries: BaseTimeSeriesGenerator | dict,
        seq_len: BaseTensorGenerator | dict,
    ) -> None:
        super().__init__()
        self._timeseries = setup_timeseries_generator(timeseries)
        self._seq_len = setup_tensor_generator(seq_len)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"timeseries": self._timeseries, "seq_len": self._seq_len}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, batch_size: int = 1, rng: Generator | None = None) -> BatchedTensorSeq:
        seq_len = int(self._seq_len.generate((1,), rng=rng).item())
        return self._timeseries.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
