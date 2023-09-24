from __future__ import annotations

__all__ = [
    "BaseTimeSeriesGenerator",
    "is_timeseries_generator_config",
    "setup_timeseries_generator",
]

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator

from objectory import AbstractFactory
from objectory.utils import is_object_config
from redcat import BatchDict

from startorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseTimeSeriesGenerator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to generate time series.

    Example usage:

    .. code-block:: pycon

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
        BatchDict(
          (value): tensor([[...]], batch_dim=0, seq_dim=1)
          (time): tensor([[...]], batch_dim=0, seq_dim=1)
        )
    """

    @abstractmethod
    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchDict:
        r"""Generates a time series.

        Args:
        ----
            seq_len (int): Specifies the sequence length.
            batch_size (int, optional): Specifies the batch size.
                Default: ``1``
            rng (``torch.Generator`` or None, optional): Specifies
                an optional random number generator. Default: ``None``

        Returns:
        -------
            ``BatchDict``: A batch of time series.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from startorch.sequence import RandUniform
            >>> from startorch.timeseries import TimeSeries
            >>> generator = TimeSeries({"value": RandUniform(), "time": RandUniform()})
            >>> generator.generate(seq_len=12, batch_size=4)
            BatchDict(
              (value): tensor([[...]], batch_dim=0, seq_dim=1)
              (time): tensor([[...]], batch_dim=0, seq_dim=1)
            )
        """


def is_timeseries_generator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseTimeSeriesGenerator``.

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
            for a ``BaseTimeSeriesGenerator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.timeseries import is_timeseries_generator_config
        >>> is_timeseries_generator_config(
        ...     {
        ...         "_target_": "startorch.timeseries.TimeSeries",
        ...         "sequences": {
        ...             "value": {"_target_": "startorch.sequence.RandUniform"},
        ...             "time": {"_target_": "startorch.sequence.RandUniform"},
        ...         },
        ...     }
        ... )
        True
    """
    return is_object_config(config, BaseTimeSeriesGenerator)


def setup_timeseries_generator(
    generator: BaseTimeSeriesGenerator | dict,
) -> BaseTimeSeriesGenerator:
    r"""Sets up a time series generator.

    The time series generator is instantiated from its configuration
    by using the ``BaseTimeSeriesGenerator`` factory function.

    Args:
    ----
        generator (``BaseTimeSeriesGenerator`` or dict): Specifies a time
            series generator or its configuration.

    Returns:
    -------
        ``BaseTimeSeriesGenerator``: A time series generator.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.timeseries import setup_timeseries_generator
        >>> setup_timeseries_generator(
        ...     {
        ...         "_target_": "startorch.timeseries.TimeSeries",
        ...         "sequences": {
        ...             "value": {"_target_": "startorch.sequence.RandUniform"},
        ...             "time": {"_target_": "startorch.sequence.RandUniform"},
        ...         },
        ...     }
        ... )
        TimeSeriesGenerator(
          (value): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
          (time): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
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
