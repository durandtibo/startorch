r"""Contain the implementation of tensor generators where the values are
sampled from a categorical distribution."""

from __future__ import annotations

__all__ = ["MultinomialTensorGenerator", "UniformCategoricalTensorGenerator"]

import math
from typing import TYPE_CHECKING

import torch
from torch import Generator, Tensor

from startorch.tensor.base import BaseTensorGenerator
from startorch.utils.weight import prepare_probabilities

if TYPE_CHECKING:
    from collections.abc import Sequence


class MultinomialTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate tensors of categorical variables
    where each value is sampled from a multinomial distribution.

    Args:
        weights: Specifies the vector of weights associated at each
            category. It must be a float tensor of shape
            ``(num_categories,)``. The weights have to be positive but
            do not need to sum to 1.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.tensor import Multinomial
    >>> generator = Multinomial(torch.ones(10))
    >>> generator
    MultinomialTensorGenerator(num_categories=10)
    >>> generator.generate(size=(4, 12))
    tensor([[...]])

    ```
    """

    def __init__(self, weights: torch.Tensor | Sequence[float]) -> None:
        super().__init__()
        self._probabilities = prepare_probabilities(weights)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_categories={self._probabilities.numel()})"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return torch.multinomial(
            self._probabilities, math.prod(size), replacement=True, generator=rng
        ).view(*size)

    @classmethod
    def create_uniform_weights(cls, num_categories: int) -> MultinomialTensorGenerator:
        r"""Instantiate the weights with a uniform pattern.

        All the categories have the same probability.
        The weight of the ``i``-th category (``w_i``) is generated
        with the rule: ``w_i = 1``

        Args:
            num_categories: Specifies the number of categories.

        Returns:
            A tensor generator where the weights of the multinomial
                distribution follow a uniform pattern.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.tensor import Multinomial
        >>> generator = Multinomial.create_uniform_weights(10)
        >>> generator
        MultinomialTensorGenerator(num_categories=10)
        >>> generator.generate(size=(4, 12))
        tensor([[...]])

        ```
        """
        return cls(weights=torch.ones(num_categories))

    @classmethod
    def create_linear_weights(
        cls,
        num_categories: int,
    ) -> MultinomialTensorGenerator:
        r"""Instantiate the weights with a linear pattern.

        The weight of the ``i``-th category (``w_i``) is generated
        with the rule: ``w_i = num_categories - i``

        Args:
            num_categories: Specifies the number of categories.

        Returns:
            A tensor generator where the weights of the multinomial
                distribution follow a linear pattern.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.tensor import Multinomial
        >>> generator = Multinomial.create_linear_weights(10)
        >>> generator
        MultinomialTensorGenerator(num_categories=10)
        >>> generator.generate(size=(4, 12))
        tensor([[...]])

        ```
        """
        return cls(
            weights=num_categories * torch.ones(num_categories) - torch.arange(num_categories)
        )

    @classmethod
    def create_exp_weights(
        cls, num_categories: int, scale: float = 0.1
    ) -> MultinomialTensorGenerator:
        r"""Instantiate the weights with an exponential pattern.

        The weight of the ``i``-th category (``w_i``) is generated with
        the rule: ``w_i = exp(-scale * i)``

        Args:
            num_categories: Specifies the number of categories.
            scale: Specifies the scale parameter that controls the
                exponential function.

        Returns:
            A tensor generator where the weights of the multinomial
                distribution follow an exponential pattern.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.tensor import Multinomial
        >>> generator = Multinomial.create_exp_weights(10)
        >>> generator
        MultinomialTensorGenerator(num_categories=10)
        >>> generator.generate(size=(4, 12))
        tensor([[...]])

        ```
        """
        return cls(weights=torch.arange(num_categories).float().mul(-scale).exp())


class UniformCategoricalTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate tensors of uniformly distributed
    categorical variables.

    All the categories have the same probability.

    Note: it is a more efficient implementation of
    ``Multinomial.generate_uniform_weights``.

    Args:
        num_categories: Specifies the number of categories.

    Raises:
        ValueError: if ``num_categories`` is negative.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.tensor import UniformCategorical
    >>> generator = UniformCategorical(10)
    >>> generator
    UniformCategoricalTensorGenerator(num_categories=10)
    >>> generator.generate(size=(4, 12))
    tensor([[...]])

    ```
    """

    def __init__(self, num_categories: int) -> None:
        super().__init__()
        if num_categories <= 0:
            msg = f"num_categories has to be greater than 0 (received: {num_categories})"
            raise ValueError(msg)
        self._num_categories = int(num_categories)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_categories={self._num_categories})"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return torch.randint(low=0, high=self._num_categories, size=size, generator=rng)
