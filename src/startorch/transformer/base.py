r"""Contain the base class to implement a transformer."""

from __future__ import annotations

__all__ = ["BaseTransformer", "is_transformer_config", "setup_transformer"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

from startorch.utils.format import str_target_object

if TYPE_CHECKING:

    import torch

logger = logging.getLogger(__name__)


class BaseTransformer(ABC, metaclass=AbstractFactory):
    r"""Define the base class to transform a batch of data.

    A child class has to implement the ``transform`` method.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Identity
    >>> transformer = Identity()
    >>> transformer
    IdentityTransformer(copy=True)
    >>> data = {"key": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    >>> out = transformer.transform(data)
    >>> out
    {'key': tensor([[1., 2., 3.],
                    [4., 5., 6.]])}

    ```
    """

    @abstractmethod
    def transform(
        self, data: dict[str, torch.Tensor], *, rng: torch.Transformer | None = None
    ) -> dict[str, torch.Tensor]:
        r"""Transform the input data.

        Args:
            data: The data to transform.
            rng: An optional random number transformer.

        Returns:
            The transformed data.

        Example usage:

        ```pycon

        >>> import torch
        >>> from startorch.transformer import Identity
        >>> transformer = Identity()
        >>> data = {'key': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
        >>> out = transformer.transform(data)
        >>> out
        {'key': tensor([[1., 2., 3.],
                        [4., 5., 6.]])}

        ```
        """


def is_transformer_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseTransformer``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration for a
            ``BaseTransformer`` object.

    Example usage:

    ```pycon

    >>> from startorch.transformer import is_transformer_config
    >>> is_transformer_config({"_target_": "startorch.transformer.Identity"})
    True

    ```
    """
    return is_object_config(config, BaseTransformer)


def setup_transformer(transformer: BaseTransformer | dict) -> BaseTransformer:
    r"""Set up a tensor transformer.

    The tensor transformer is instantiated from its configuration by
    using the ``BaseTransformer`` factory function.

    Args:
        transformer: A tensor transformer or its configuration.

    Returns:
        A tensor transformer.

    Example usage:

    ```pycon

    >>> from startorch.transformer import setup_transformer
    >>> setup_transformer({"_target_": "startorch.transformer.Identity"})
    IdentityTransformer(copy=True)

    ```
    """
    if isinstance(transformer, dict):
        logger.info(
            "Initializing a tensor transformer from its configuration... "
            f"{str_target_object(transformer)}"
        )
        transformer = BaseTransformer.factory(**transformer)
    if not isinstance(transformer, BaseTransformer):
        logger.warning(f"transformer is not a `BaseTransformer` (received: {type(transformer)})")
    return transformer
