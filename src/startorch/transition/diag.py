r"""Contain diagonal based transition matrix generators."""

from __future__ import annotations

__all__ = ["DiagonalTransitionGenerator", "PermuteDiagonalTransitionGenerator"]

import torch

from startorch.transition import BaseTransitionGenerator


class DiagonalTransitionGenerator(BaseTransitionGenerator):
    r"""Implement a simple diagonal transition matrix generator.

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

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def generate(
        self,
        n: int,
        rng: torch.Generator | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return torch.diag(torch.ones(n))


class PermuteDiagonalTransitionGenerator(BaseTransitionGenerator):
    r"""Implement a simple diagonal transition matrix generator.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transition import PermuteDiagonal
    >>> generator = PermuteDiagonal()
    >>> generator
    PermuteDiagonalTransitionGenerator()
    >>> generator.generate(n=6)
    tensor([...]])

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def generate(
        self,
        n: int,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.diag(torch.ones(n))[torch.randperm(n, generator=rng)]
