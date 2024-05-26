r"""Contain transition matrix generators."""

from __future__ import annotations

__all__ = ["BaseTransitionGenerator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory

if TYPE_CHECKING:
    import torch


class BaseTransitionGenerator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to generate a transition matrix.

    A child class has to implement the ``generate`` method.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transition import Diagonal
    >>> generator = Diagonal()
    >>> generator
    DiagonalTransitionGenerator()
    >>> generator.generate(n=6)
    tensor([[1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.]])

    ```
    """

    @abstractmethod
    def generate(self, n: int, rng: torch.Generator | None = None) -> torch.Tensor:
        r"""Return a transition matrix.

        Args:
            n: The size of the transition matrix.
            rng: An optional random number generator.

        Returns:
            The generated transition matrix of shape ``(n, n)`` and
                data type float.

        Example usage:

        ```pycon

        >>> import torch
        >>> from startorch.transition import Diagonal
        >>> generator = Diagonal()
        >>> generator.generate(n=6)
        tensor([[1., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 1.]])

        ```
        """
