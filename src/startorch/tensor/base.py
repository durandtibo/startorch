from __future__ import annotations

__all__ = ["BaseTensorGenerator", "setup_tensor_generator"]

import logging
from abc import ABC, abstractmethod

from objectory import AbstractFactory
from torch import Generator, Tensor

from startorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseTensorGenerator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to generate tensor.

    A child class has to implement the ``generate`` method.
    """

    @abstractmethod
    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        r"""Generates a tensor.

        Args:
        ----
            size (tuple): Specifies the size of the tensor to generate.
            rng (``torch.Generator`` or None, optional): Specifies
                an optional random number generator. Default: ``None``

        Returns:
        -------
            ``torch.Tensor``: The generated tensor with the specified
                size.
        """


def setup_tensor_generator(generator: BaseTensorGenerator | dict) -> BaseTensorGenerator:
    r"""Sets up a tensor generator.

    The tensor generator is instantiated from its configuration by
    using the ``BaseTensorGenerator`` factory function.

    Args:
    ----
        generator (``BaseTensorGenerator`` or dict): Specifies a
            tensor generator or its configuration.

    Returns:
    -------
        ``BaseTensorGenerator``: A tensor generator.
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing a tensor generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BaseTensorGenerator.factory(**generator)
    return generator
