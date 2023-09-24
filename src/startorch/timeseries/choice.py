from __future__ import annotations

__all__ = ["MultinomialChoiceTimeSeriesGenerator"]

from collections.abc import Generator, Sequence

import torch
from coola.utils import str_indent
from redcat import BatchDict

from startorch.timeseries.base import (
    BaseTimeSeriesGenerator,
    setup_timeseries_generator,
)
from startorch.utils.format import str_weighted_modules
from startorch.utils.weight import prepare_weighted_generators


class MultinomialChoiceTimeSeriesGenerator(BaseTimeSeriesGenerator):
    r"""Implements a time series generator that select a time series
    generator at each batch.

    This time series generator is used to generate time series with
    different generation processes. The user can specify a list of
    time series generators with an associated weight. The weight is
    used to sample the time series generator with a multinomial
    distribution. Higher weight means that the time series generator
    has a higher probability to be selected at each batch.
    Each dictionary in the ``generators`` input should have the
    following items:

        - a key ``'generator'`` which indicates the time series
            generator or its configuration.
        - an optional key ``'weight'`` with a float value which
            indicates the weight of the time series generator.
            If this key is absent, the weight is set to ``1.0``.

    Args:
    ----
        generators (tuple or list): Specifies the time series
            generators and their weights. See above to learn
            about the expected format.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.timeseries import MultinomialChoice, TimeSeries
        >>> from startorch.sequence import RandUniform, RandNormal
        >>> generator = MultinomialChoice(
        ...     (
        ...         {
        ...             "weight": 2.0,
        ...             "generator": TimeSeries({"value": RandUniform(), "time": RandUniform()}),
        ...         },
        ...         {
        ...             "weight": 1.0,
        ...             "generator": TimeSeries({"value": RandNormal(), "time": RandNormal()}),
        ...         },
        ...     )
        ... )
        >>> generator
        MultinomialChoiceTimeSeriesGenerator(
          (0) [weight=2.0] TimeSeriesGenerator(
              (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
              (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
            )
          (1) [weight=1.0] TimeSeriesGenerator(
              (value): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
              (time): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
            )
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        BatchDict(
          (value): tensor([[...]], batch_dim=0, seq_dim=1)
          (time): tensor([[...]], batch_dim=0, seq_dim=1)
        )
    """

    def __init__(self, generators: Sequence[dict[str, BaseTimeSeriesGenerator | dict]]) -> None:
        super().__init__()
        generators, weights = prepare_weighted_generators(generators)
        self._generators = tuple(setup_timeseries_generator(generator) for generator in generators)
        self._weights = torch.as_tensor(weights, dtype=torch.float)

    def __repr__(self) -> str:
        args = str_indent(str_weighted_modules(modules=self._generators, weights=self._weights))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchDict:
        index = torch.multinomial(self._weights, num_samples=1, generator=rng).item()
        return self._generators[index].generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
