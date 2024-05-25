r"""Contain the implementation of example generator that generates
sequences from a time series generator."""

from __future__ import annotations

__all__ = ["TimeSeriesExampleGenerator"]


from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.example.base import BaseExampleGenerator
from startorch.tensor.base import setup_tensor_generator
from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch

    from startorch.tensor.base import BaseTensorGenerator


class TimeSeriesExampleGenerator(BaseExampleGenerator):
    r"""Implement an example generator to generate time series.

    Args:
        generators: A time series generator or its
            configuration.
        seq_len: The sequence length sampler or its
            configuration. This sampler is used to sample the
            sequence length at each batch.

    Example usage:

    ```pycon

    >>> from startorch.example import TimeSeriesExampleGenerator
    >>> from startorch.timeseries import SequenceTimeSeriesGenerator
    >>> from startorch.sequence import Periodic, RandUniform
    >>> from startorch.tensor import RandInt
    >>> generator = TimeSeriesExampleGenerator(
    ...     generators=SequenceTimeSeriesGenerator(
    ...         {"value": RandUniform(), "time": RandUniform()}
    ...     ),
    ...     seq_len=RandInt(2, 5),
    ... )
    >>> generator
    TimeSeriesExampleGenerator(
      (generators): SequenceTimeSeriesGenerator(
          (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
          (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
      (seq_len): RandIntTensorGenerator(low=2, high=5)
    )
    >>> generator.generate(batch_size=10)
    {'value': tensor([[...]]), 'time': tensor([[...]])}

    ```
    """

    def __init__(
        self,
        generators: BaseTimeSeriesGenerator | dict,
        seq_len: BaseTensorGenerator | dict,
    ) -> None:
        super().__init__()
        self._generators = setup_timeseries_generator(generators)
        self._seq_len = setup_tensor_generator(seq_len)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"generators": self._generators, "seq_len": self._seq_len}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        seq_len = int(self._seq_len.generate((1,), rng=rng).item())
        return self._generators.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
