r"""Contain the base class to implement a transformer."""

from __future__ import annotations

__all__ = ["BaseTensorTransformer", "BaseTransformer", "is_transformer_config", "setup_transformer"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

from startorch.transformer.utils import add_item, check_input_keys
from startorch.utils.format import str_target_object

if TYPE_CHECKING:
    from collections.abc import Hashable

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
        self, data: dict[Hashable, torch.Tensor], *, rng: torch.Transformer | None = None
    ) -> dict[Hashable, torch.Tensor]:
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
        >>> data = {"key": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
        >>> out = transformer.transform(data)
        >>> out
        {'key': tensor([[1., 2., 3.],
                        [4., 5., 6.]])}

        ```
        """


class BaseTensorTransformer(BaseTransformer):
    r"""Define the base class to transform a tensor.

    Args:
        input: The key that contains the input tensor.
        output: The key that contains the output tensor.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import LookupTable
    >>> transformer = LookupTable(
    ...     weights=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), index="index", output="output"
    ... )
    >>> transformer
    LookupTableTransformer(size=6, index=index, output=output, exist_ok=False)
    >>> data = {"index": torch.tensor([[1, 2, 3], [4, 0, 2]])}
    >>> out = transformer.transform(data)
    >>> out
    {'index': tensor([[1, 2, 3], [4, 0, 2]]), 'output': tensor([[4., 3., 2.], [1., 5., 3.]])}

    ```
    """

    def __init__(self, input: str, output: str, exist_ok: bool = False) -> None:  # noqa: A002
        super().__init__()
        self._input = input
        self._output = output
        self._exist_ok = exist_ok

    # def __repr__(self) -> str:
    #     return (
    #         f"{self.__class__.__qualname__}(input={self._input}, output={self._output}, "
    #         f"exist_ok={self._exist_ok})"
    #     )

    def transform(
        self,
        data: dict[Hashable, torch.Tensor],
        *,
        rng: torch.Transformer | None = None,
    ) -> dict[Hashable, torch.Tensor]:
        check_input_keys(data, keys=[self._input])
        data = data.copy()
        add_item(
            data,
            key=self._output,
            value=self._transform(tensor=data[self._input], rng=rng),
            exist_ok=self._exist_ok,
        )
        return data

    @abstractmethod
    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,
    ) -> torch.Tensor:
        r"""Transform a tensor.

        Args:
            tensor: The tensor to transform.
            rng: An optional random number transformer.

        Returns:
            The transformed tensor.
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
