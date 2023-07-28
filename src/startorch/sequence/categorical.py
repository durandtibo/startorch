from __future__ import annotations

__all__ = ["MultinomialSequenceGenerator", "UniformCategoricalSequenceGenerator"]

import math

import torch
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator
from startorch.utils.conversion import to_tuple


class MultinomialSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequences of categorical variables
    where each value is sampled from a multinomial distribution.

    Args:
    ----
        weights (``torch.Tensor`` of shape ``(num_categories,)`` and
            type float): Specifies the vector of weights associated
            at each category. The weights have to be positive but do
            not need to sum to 1.
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import Multinomial
        >>> generator = Multinomial(weights=torch.ones(5))
        >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        weights: torch.Tensor,
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
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            torch.multinomial(
                self._probabilities,
                batch_size * seq_len * math.prod(self._feature_size),
                replacement=True,
                generator=rng,
            ).view(batch_size, seq_len, *self._feature_size)
        )

    @classmethod
    def create_uniform_weights(cls, num_categories: int) -> MultinomialSequenceGenerator:
        r"""Initializes the weights with a uniform pattern.

        All the categories have the same probability.
        The weight of the ``i``-th category (``w_i``) is generated
        with the rule: ``w_i = 1``

        Args:
        ----
            num_categories (int): Specifies the number of categories.

        Returns:
        -------
            ``MultinomialSequenceGenerator``: A sequence generator where
                the weights of the multinomial distribution follow
                a uniform pattern.
        """
        return cls(weights=torch.ones(num_categories))

    @classmethod
    def create_linear_weights(cls, num_categories: int) -> MultinomialSequenceGenerator:
        r"""Initializes the weights with a linear pattern.

        The weight of the ``i``-th category (``w_i``) is generated
        with the rule: ``w_i = num_categories - i``

        Args:
        ----
            num_categories (int): Specifies the number of categories.

        Returns:
        -------
            ``MultinomialSequenceGenerator``: A sequence generator where
                the weights of the multinomial distribution follow a
                linear pattern.
        """
        return cls(
            weights=num_categories * torch.ones(num_categories) - torch.arange(num_categories)
        )

    @classmethod
    def create_exp_weights(
        cls, num_categories: int, scale: float = 0.1
    ) -> MultinomialSequenceGenerator:
        r"""Initializes the weights with an exponential pattern.

        The weight of the ``i``-th category (``w_i``) is generated with
        the rule: ``w_i = exp(-scale * i)``

        Args:
        ----
            num_categories (int): Specifies the number of categories.
            scale (float, optional): Specifies the scale parameter
                that controls the exponential function.
                Default: ``0.1``

        Returns:
        -------
            ``MultinomialSequenceGenerator``: A sequence generator where
                the weights of the multinomial distribution follow
                an exponential pattern.
        """
        return cls(weights=torch.arange(num_categories).float().mul(-scale).exp())


def prepare_probabilities(weights: torch.Tensor) -> torch.Tensor:
    r"""Converts un-normalized positive weights to probabilities.

    Args:
    ----
        weights (``torch.Tensor`` of shape ``(num_categories,)`` and
            type float): Specifies the vector of weights associated to
            each category. The weights have to be positive.

    Returns:
    -------
        ``torch.Tensor`` of type float and shape ``(num_categories,)``:
            The vector of probability associated at each category.

    Raises:
    ------
        ValueError if the weights are not valid.
    """
    if weights.ndim != 1:
        raise ValueError(f"weights has to be a 1D tensor (received a {weights.ndim}D tensor)")
    if weights.min() < 0:
        raise ValueError(
            f"The values in weights have to be positive (min: {weights.min()}  weights: {weights})"
        )
    if weights.sum() == 0:
        raise ValueError(
            f"The sum of the weights has to be greater than 0 (sum: {weights.sum()}  "
            f"weights: {weights})"
        )
    return weights.float() / weights.sum()


class UniformCategoricalSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequences of uniformly distributed
    categorical variables.

    All the categories have the same probability.

    Note: it is a more efficient implementation of
    ``Multinomial.generate_uniform_weights``.

    Args:
    ----
        num_categories (int): Specifies the number of categories.
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``tuple()``

    Raises:
    ------
        ValueError if ``num_categories`` is negative.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import UniformCategorical
        >>> generator = UniformCategorical(10)
        >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        num_categories: int,
        feature_size: tuple[int, ...] | list[int] | int = tuple(),
    ) -> None:
        super().__init__()
        if num_categories <= 0:
            raise ValueError(
                f"num_categories has to be greater than 0 (received: {num_categories})"
            )
        self._num_categories = int(num_categories)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_categories={self._num_categories}, "
            f"feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            torch.randint(
                low=0,
                high=self._num_categories,
                size=(batch_size, seq_len) + self._feature_size,
                generator=rng,
            )
        )
