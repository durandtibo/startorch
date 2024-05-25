r"""Contain the base class to implement a time series generator."""

from __future__ import annotations

__all__ = [
    "BaseTimeSeriesGenerator",
    "is_timeseries_generator_config",
    "setup_timeseries_generator",
]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

from startorch.utils.format import str_target_object

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch

logger = logging.getLogger(__name__)


class BaseTimeSeriesGenerator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a time series generator.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import RandUniform
    >>> from startorch.timeseries import TimeSeries
    >>> generator = TimeSeries({"value": RandUniform(), "time": RandUniform()})
    >>> generator
    TimeSeriesGenerator(
      (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
      (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    {'value': tensor([[...]]), 'time': tensor([[...]])}

    ```
    """

    @abstractmethod
    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        r"""Generate a time series.

        Args:
            seq_len: The sequence length.
            batch_size: The batch size.
            rng: An optional random number generator.

        Returns:
            A batch of time series.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.sequence import RandUniform
        >>> from startorch.timeseries import TimeSeries
        >>> generator = TimeSeries({"value": RandUniform(), "time": RandUniform()})
        >>> generator.generate(seq_len=12, batch_size=4)
        {'value': tensor([[...]]), 'time': tensor([[...]])}

        ```
        """


def is_timeseries_generator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseTimeSeriesGenerator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseTimeSeriesGenerator`` object.

    Example usage:

    ```pycon

    >>> from startorch.timeseries import is_timeseries_generator_config
    >>> is_timeseries_generator_config(
    ...     {
    ...         "_target_": "startorch.timeseries.TimeSeries",
    ...         "generators": {
    ...             "value": {"_target_": "startorch.sequence.RandUniform"},
    ...             "time": {"_target_": "startorch.sequence.RandUniform"},
    ...         },
    ...     }
    ... )
    True

    ```
    """
    return is_object_config(config, BaseTimeSeriesGenerator)


def setup_timeseries_generator(
    generator: BaseTimeSeriesGenerator | dict,
) -> BaseTimeSeriesGenerator:
    r"""Set up a time series generator.

    The time series generator is instantiated from its configuration
    by using the ``BaseTimeSeriesGenerator`` factory function.

    Args:
        generator: A time series generator or its
            configuration.

    Returns:
        A time series generator.

    Example usage:

    ```pycon

    >>> from startorch.timeseries import setup_timeseries_generator
    >>> setup_timeseries_generator(
    ...     {
    ...         "_target_": "startorch.timeseries.TimeSeries",
    ...         "generators": {
    ...             "value": {"_target_": "startorch.sequence.RandUniform"},
    ...             "time": {"_target_": "startorch.sequence.RandUniform"},
    ...         },
    ...     }
    ... )
    TimeSeriesGenerator(
      (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
      (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )

    ```
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing a time-series generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BaseTimeSeriesGenerator.factory(**generator)
    if not isinstance(generator, BaseTimeSeriesGenerator):
        logger.warning(
            f"generator is not a `BaseTimeSeriesGenerator` (received: {type(generator)})"
        )
    return generator
