r"""Contain the implementation of a time series generator that generates
a batch of time series, and then transforms them."""

from __future__ import annotations

__all__ = ["TransformTimeSeriesGenerator"]


from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)
from startorch.transformer.base import BaseTransformer, setup_transformer

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch


class TransformTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implement a time series generator that generates a batch of time
    series, and then transforms them.

    Args:
        generator: The time series generator or its configuration.
        transformer: The data transformer or its configuration.

    Example usage:

    ```pycon

    >>> from startorch.timeseries import TransformTimeSeriesGenerator, SequenceTimeSeries
    >>> from startorch.transformer import TensorTransformer
    >>> from startorch.sequence import RandUniform
    >>> from startorch.tensor.transformer import Abs
    >>> generator = TransformTimeSeriesGenerator(
    ...     generator=SequenceTimeSeries({"time": RandUniform(), "value": RandUniform()}),
    ...     transformer=TensorTransformer(
    ...         transformer=Abs(), input="value", output="value_transformed"
    ...     ),
    ... )
    >>> generator
    TransformTimeSeriesGenerator(
      (generator): SequenceTimeSeriesGenerator(
          (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
          (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
      (transformer): TensorTransformer(
          (transformer): AbsTensorTransformer()
          (input): value
          (output): value_transformed
          (exist_ok): False
        )
    )
    >>> generator.generate(batch_size=4, seq_len=12)
    {'time': tensor([[[...]]]), 'value': tensor([[[...]]]), 'value_transformed': tensor([[[...]]])}

    ```
    """

    def __init__(
        self,
        generator: BaseTimeSeriesGenerator | dict,
        transformer: BaseTransformer | dict,
    ) -> None:
        super().__init__()
        self._generator = setup_timeseries_generator(generator)
        self._transformer = setup_transformer(transformer)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping({"generator": self._generator, "transformer": self._transformer})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        data = self._generator.generate(batch_size=batch_size, seq_len=seq_len, rng=rng)
        return self._transformer.transform(data, rng=rng)
