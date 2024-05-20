r"""Contain the base class to implement a tensor transformer."""

from __future__ import annotations

__all__ = ["BaseTensorTransformer", "is_tensor_transformer_config", "setup_tensor_transformer"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

from startorch.utils.format import str_target_object

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

logger = logging.getLogger(__name__)


class BaseTensorTransformer(ABC, metaclass=AbstractFactory):
    r"""Define the base class to transform a tensor.

    A child class has to implement the ``transform`` method.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer.tensor import Identity
    >>> transformer = Identity()
    >>> transformer
    IdentityTensorTransformer(copy=True)
    >>> tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> out = transformer.transform([tensor])
    >>> out
    tensor([[1., 2., 3.],
            [4., 5., 6.]])

    ```
    """

    @abstractmethod
    def transform(
        self, tensors: Sequence[torch.Tensor], *, rng: torch.Transformer | None = None
    ) -> torch.Tensor:
        r"""Generate a tensor.

        Args:
            tensors: The input tensors used to generate the output transform.
            rng: An optional random number transformer.

        Returns:
            The transformed tensor.

        Example usage:

        ```pycon

        >>> import torch
        >>> from startorch.transformer.tensor import Identity
        >>> transformer = Identity()
        >>> tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> out = transformer.transform([tensor])
        >>> out
        tensor([[1., 2., 3.],
                [4., 5., 6.]])

        ```
        """


def is_tensor_transformer_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseTensorTransformer``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseTensorTransformer`` object.

    Example usage:

    ```pycon
    >>> from startorch.transformer.tensor import is_tensor_transformer_config
    >>> is_tensor_transformer_config({"_target_": "startorch.transformer.tensor.Identity"})
    True

    ```
    """
    return is_object_config(config, BaseTensorTransformer)


def setup_tensor_transformer(transformer: BaseTensorTransformer | dict) -> BaseTensorTransformer:
    r"""Set up a tensor transformer.

    The tensor transformer is instantiated from its configuration by
    using the ``BaseTensorTransformer`` factory function.

    Args:
        transformer: A tensor transformer or its configuration.

    Returns:
        A tensor transformer.

    Example usage:

    ```pycon
    >>> from startorch.transformer.tensor import setup_tensor_transformer
    >>> setup_tensor_transformer({"_target_": "startorch.transformer.tensor.Identity"})
    IdentityTensorTransformer(copy=True)

    ```
    """
    if isinstance(transformer, dict):
        logger.info(
            "Initializing a tensor transformer from its configuration... "
            f"{str_target_object(transformer)}"
        )
        transformer = BaseTensorTransformer.factory(**transformer)
    if not isinstance(transformer, BaseTensorTransformer):
        logger.warning(
            f"transformer is not a `BaseTensorTransformer` (received: {type(transformer)})"
        )
    return transformer
