r"""Contain the implementation of sequence generators where the values
are sampled from a categorical distribution."""

from __future__ import annotations

__all__ = ["MultinomialSequenceGenerator", "UniformCategoricalSequenceGenerator"]

import math
from typing import TYPE_CHECKING

import torch

from startorch.sequence.base import BaseSequenceGenerator
from startorch.utils.conversion import to_tuple
from startorch.utils.weight import prepare_probabilities

if TYPE_CHECKING:
    from collections.abc import Sequence


class MultinomialSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequences of categorical variables
    where each value is sampled from a multinomial distribution.

    Args:
        weights (``torch.Tensor`` of shape ``(num_categories,)`` and
            type float): Specifies the vector of weights associated
            at each category. The weights have to be positive but do
            not need to sum to 1.
        feature_size: Specifies the feature size.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import Multinomial
    >>> generator = Multinomial(weights=torch.ones(5))
    >>> generator
    MultinomialSequenceGenerator(num_categories=5, feature_size=(1,))
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        weights: torch.Tensor | Sequence[float],
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        self._probabilities = prepare_probabilities(weights)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_categories={self._probabilities.numel()}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return torch.multinomial(
            self._probabilities,
            batch_size * seq_len * math.prod(self._feature_size),
            replacement=True,
            generator=rng,
        ).view(batch_size, seq_len, *self._feature_size)

    @classmethod
    def create_uniform_weights(cls, num_categories: int) -> MultinomialSequenceGenerator:
        r"""Instantiate the weights with a uniform pattern.

        All the categories have the same probability.
        The weight of the ``i``-th category (``w_i``) is generated
        with the rule: ``w_i = 1``

        Args:
            num_categories: Specifies the number of categories.

        Returns:
            A sequence generator where the weights of the multinomial
                distribution follow a uniform pattern.
        """
        return cls(weights=torch.ones(num_categories))

    @classmethod
    def create_linear_weights(cls, num_categories: int) -> MultinomialSequenceGenerator:
        r"""Instantiate the weights with a linear pattern.

        The weight of the ``i``-th category (``w_i``) is generated
        with the rule: ``w_i = num_categories - i``

        Args:
            num_categories: Specifies the number of categories.

        Returns:
            A sequence generator where the weights of the multinomial
                distribution follow a linear pattern.
        """
        return cls(
            weights=num_categories * torch.ones(num_categories) - torch.arange(num_categories)
        )

    @classmethod
    def create_exp_weights(
        cls, num_categories: int, scale: float = 0.1
    ) -> MultinomialSequenceGenerator:
        r"""Instantiate the weights with an exponential pattern.

        The weight of the ``i``-th category (``w_i``) is generated with
        the rule: ``w_i = exp(-scale * i)``

        Args:
            num_categories: Specifies the number of categories.
            scale: Specifies the scale parameter that controls the
                exponential function.

        Returns:
            A sequence generator where the weights of the multinomial
                distribution follow an exponential pattern.
        """
        return cls(weights=torch.arange(num_categories).float().mul(-scale).exp())


class UniformCategoricalSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequences of uniformly distributed
    categorical variables.

    All the categories have the same probability.

    Note:
        It is a more efficient implementation of
        ``Multinomial.generate_uniform_weights``.

    Args:
        num_categories: Specifies the number of categories.
        feature_size: Specifies the feature size.

    Raises:
        ValueError: if ``num_categories`` is negative.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import UniformCategorical
    >>> generator = UniformCategorical(10)
    >>> generator
    UniformCategoricalSequenceGenerator(num_categories=10, feature_size=())
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        num_categories: int,
        feature_size: tuple[int, ...] | list[int] | int = (),
    ) -> None:
        super().__init__()
        if num_categories <= 0:
            msg = f"num_categories has to be greater than 0 (received: {num_categories})"
            raise ValueError(msg)
        self._num_categories = int(num_categories)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_categories={self._num_categories}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self._num_categories,
            size=(batch_size, seq_len, *self._feature_size),
            generator=rng,
        )
