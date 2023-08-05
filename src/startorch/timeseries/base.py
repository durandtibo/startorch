from __future__ import annotations

__all__ = ["BaseTimeSeriesGenerator", "setup_timeseries_generator"]

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator

from objectory import AbstractFactory
from redcat import BatchDict

from startorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseTimeSeriesGenerator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to generate time series."""

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
        """


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
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing a time series generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BaseTimeSeriesGenerator.factory(**generator)
    return generator
