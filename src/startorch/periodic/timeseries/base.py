from __future__ import annotations

__all__ = [
    "BasePeriodicTimeSeriesGenerator",
    "is_periodic_timeseries_generator_config",
    "setup_periodic_timeseries_generator",
]

import logging
from abc import ABC, abstractmethod

from objectory import AbstractFactory
from objectory.utils import is_object_config
from redcat import BatchDict
from torch import Generator

from startorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BasePeriodicTimeSeriesGenerator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to generate periodic time series.

    A child class has to implement the ``generate`` method.

    Example usage:

    .. code-block:: pycon

        Example usage:

    .. code-block:: pycon

        >>> from startorch.periodic.timeseries import Repeat
        >>> from startorch.timeseries import TimeSeries
        >>> from startorch.sequence import RandUniform
        >>> generator = Repeat(TimeSeries({"value": RandUniform(), "time": RandUniform()}))
        >>> generator
        RepeatPeriodicTimeSeriesGenerator(
          (generator): TimeSeriesGenerator(
              (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
              (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
            )
        )
        >>> generator.generate(seq_len=12, period=4, batch_size=4)
        BatchDict(
          (value): tensor([[...]], batch_dim=0, seq_dim=1)
          (time): tensor([[...]], batch_dim=0, seq_dim=1)
        )
    """

    @abstractmethod
    def generate(
        self, seq_len: int, period: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchDict:
        r"""Generates a batch of periodic time series.

        All the time series in the batch have the same length.

        Args:
        ----
            seq_len (int): Specifies the sequence length.
            period (int): Specifies the period.
            batch_size (int, optional): Specifies the batch size.
                Default: ``1``
            rng (``torch.Generator`` or None, optional): Specifies
                an optional random number generator. Default: ``None``

        Returns:
        -------
            ``BatchDict``: A batch of periodic time series.

        Example usage:

        .. code-block:: pycon

            >>> from startorch.periodic.timeseries import Repeat
            >>> from startorch.timeseries import TimeSeries
            >>> from startorch.sequence import RandUniform
            >>> generator = Repeat(TimeSeries({"value": RandUniform(), "time": RandUniform()}))
            >>> generator.generate(seq_len=12, period=4, batch_size=4)
            BatchDict(
              (value): tensor([[...]], batch_dim=0, seq_dim=1)
              (time): tensor([[...]], batch_dim=0, seq_dim=1)
            )
        """


def is_periodic_timeseries_generator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BasePeriodicTimeSeriesGenerator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BasePeriodicTimeSeriesGenerator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.periodic.timeseries import is_periodic_timeseries_generator_config
        >>> is_periodic_timeseries_generator_config(
        ...     {
        ...         "_target_": "startorch.periodic.timeseries.Repeat",
        ...         "generator": {
        ...             "_target_": "startorch.timeseries.TimeSeries",
        ...             "sequences": {
        ...                 "value": {"_target_": "startorch.sequence.RandUniform"},
        ...                 "time": {"_target_": "startorch.sequence.RandUniform"},
        ...             },
        ...         },
        ...     }
        ... )
        True
    """
    return is_object_config(config, BasePeriodicTimeSeriesGenerator)


def setup_periodic_timeseries_generator(
    generator: BasePeriodicTimeSeriesGenerator | dict,
) -> BasePeriodicTimeSeriesGenerator:
    r"""Sets up a periodic time series generator.

    The time series generator is instantiated from its configuration by
    using the ``BasePeriodicTimeSeriesGenerator`` factory function.

    Args:
    ----
        generator (``BasePeriodicTimeSeriesGenerator`` or dict): Specifies a
            periodic time series generator or its configuration.

    Returns:
    -------
        ``BasePeriodicTimeSeriesGenerator``: A periodic time series generator.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.periodic.timeseries import setup_periodic_timeseries_generator
        >>> setup_periodic_timeseries_generator(
        ...     {
        ...         "_target_": "startorch.periodic.timeseries.Repeat",
        ...         "generator": {
        ...             "_target_": "startorch.timeseries.TimeSeries",
        ...             "sequences": {
        ...                 "value": {"_target_": "startorch.sequence.RandUniform"},
        ...                 "time": {"_target_": "startorch.sequence.RandUniform"},
        ...             },
        ...         },
        ...     }
        ... )
        RepeatPeriodicTimeSeriesGenerator(
          (generator): TimeSeriesGenerator(
              (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
              (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
            )
        )
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing a periodic time series generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BasePeriodicTimeSeriesGenerator.factory(**generator)
    if not isinstance(generator, BasePeriodicTimeSeriesGenerator):
        logger.warning(
            f"generator is not a `BasePeriodicTimeSeriesGenerator` (received: {type(generator)})"
        )
    return generator
