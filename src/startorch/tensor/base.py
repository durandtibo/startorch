r"""Contain the base class to implement a tensor generator."""

from __future__ import annotations

__all__ = ["BaseTensorGenerator", "is_tensor_generator_config", "setup_tensor_generator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

from startorch.utils.format import str_target_object

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class BaseTensorGenerator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to generate tensor.

    A child class has to implement the ``generate`` method.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.tensor import RandUniform
    >>> generator = RandUniform()
    >>> generator
    RandUniformTensorGenerator(low=0.0, high=1.0)
    >>> generator.generate(size=(4, 12))
    tensor([[...]])

    ```
    """

    @abstractmethod
    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        r"""Generate a tensor.

        Args:
            size: Specifies the size of the tensor to generate.
            rng: Specifies an optional random number generator.

        Returns:
            The generated tensor with the specified size.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.tensor import RandUniform
        >>> generator = RandUniform()
        >>> generator.generate(size=(4, 12))
        tensor([[...]])

        ```
        """


def is_tensor_generator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseTensorGenerator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: Specifies the configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseTensorGenerator`` object.

    Example usage:

    ```pycon

    >>> from startorch.tensor import is_tensor_generator_config
    >>> is_tensor_generator_config({"_target_": "startorch.tensor.RandUniform"})
    True

    ```
    """
    return is_object_config(config, BaseTensorGenerator)


def setup_tensor_generator(generator: BaseTensorGenerator | dict) -> BaseTensorGenerator:
    r"""Set up a tensor generator.

    The tensor generator is instantiated from its configuration by
    using the ``BaseTensorGenerator`` factory function.

    Args:
        generator: Specifies a tensor generator or its configuration.

    Returns:
        A tensor generator.

    Example usage:

    ```pycon

    >>> from startorch.tensor import setup_tensor_generator
    >>> setup_tensor_generator({"_target_": "startorch.tensor.RandUniform"})
    RandUniformTensorGenerator(low=0.0, high=1.0)

    ```
    """
    if isinstance(generator, dict):
        logger.info(
            "Initializing a tensor generator from its configuration... "
            f"{str_target_object(generator)}"
        )
        generator = BaseTensorGenerator.factory(**generator)
    if not isinstance(generator, BaseTensorGenerator):
        logger.warning(f"generator is not a `BaseTensorGenerator` (received: {type(generator)})")
    return generator
