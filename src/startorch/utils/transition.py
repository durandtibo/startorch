r"""Contain transition matrix generators."""

from __future__ import annotations

__all__ = ["BaseTransitionGenerator", "DiagonalTransitionGenerator"]

from abc import ABC, abstractmethod

import torch
from objectory import AbstractFactory


class BaseTransitionGenerator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to generate a transition matrix.

    A child class has to implement the ``generate`` method.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.utils.transition import DiagonalTransitionGenerator
    >>> generator = DiagonalTransitionGenerator()
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
        >>> from startorch.utils.transition import DiagonalTransitionGenerator
        >>> generator = DiagonalTransitionGenerator()
        >>> generator.generate(n=6)
        tensor([[1., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 1.]])

        ```
        """


class DiagonalTransitionGenerator(BaseTransitionGenerator):
    r"""Implement a simple diagonal transition matrix generator.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.utils.transition import DiagonalTransitionGenerator
    >>> generator = DiagonalTransitionGenerator()
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

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def generate(
        self,
        n: int,
        rng: torch.Generator | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return torch.diag(torch.ones(n))
