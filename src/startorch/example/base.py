from __future__ import annotations

__all__ = [
    "BaseExampleGenerator",
    "is_example_generator_config",
    "setup_example_generator",
]

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator

from objectory import AbstractFactory
from objectory.utils import is_object_config
from redcat import BatchDict

from startorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseExampleGenerator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to generate time series.

    Example usage:

    .. code-block:: pycon

        >>> import torch  # TODO
    """

    @abstractmethod
    def generate(self, batch_size: int = 1, rng: Generator | None = None) -> BatchDict:
        r"""Generates a time series.

        Args:
        ----
            batch_size (int, optional): Specifies the batch size.
                Default: ``1``
            rng (``torch.Generator`` or None, optional): Specifies
                an optional random number generator. Default: ``None``

        Returns:
        -------
            ``BatchDict``: A batch of time series.

        Example usage:

        .. code-block:: pycon

            >>> import torch  # TODO
        """


def is_example_generator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseExampleGenerator``.

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
            for a ``BaseExampleGenerator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import is_example_generator_config  # TODO
    """
    return is_object_config(config, BaseExampleGenerator)


def setup_example_generator(
    generator: BaseExampleGenerator | dict,
) -> BaseExampleGenerator:
    r"""Sets up a time series generator.

    The time series generator is instantiated from its configuration
    by using the ``BaseExampleGenerator`` factory function.

    Args:
    ----
        generator (``BaseExampleGenerator`` or dict): Specifies a time
            series generator or its configuration.

    Returns:
    -------
        ``BaseExampleGenerator``: A time series generator.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.example import setup_example_generator  # TODO
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing an example generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BaseExampleGenerator.factory(**generator)
    if not isinstance(generator, BaseExampleGenerator):
        logger.warning(f"generator is not a `BaseExampleGenerator` (received: {type(generator)})")
    return generator
