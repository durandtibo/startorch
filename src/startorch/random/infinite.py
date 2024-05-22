r"""Contain functions to sample values from continuous uni-variate
distributions with infinite support."""

from __future__ import annotations

__all__ = ["cauchy", "normal", "rand_cauchy", "rand_normal"]


import torch


def rand_cauchy(
    size: list[int] | tuple[int, ...],
    loc: float = 0.0,
    scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Create a sequence of continuous variables sampled from a Cauchy
    distribution.

    Args:
        size: The tensor shape.
        loc: The location/median of the Cauchy distribution.
        scale: The scale of the Cauchy distribution.
            This value has to be greater than 0.
        generator: An optional random generator.

    Returns:
        A tensor filled with values sampled from a Cauchy distribution.

    Raises:
        ValueError: if the ``scale`` parameter is not valid.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.random import rand_cauchy
    >>> rand_cauchy((2, 3), loc=1.0, scale=2.0)
    tensor([[...]])

    ```
    """
    if scale <= 0:
        msg = f"scale has to be greater than 0 (received: {scale})"
        raise ValueError(msg)
    sequence = torch.zeros(*size, dtype=torch.float)
    sequence.cauchy_(median=loc, sigma=scale, generator=generator)
    return sequence


def cauchy(
    loc: torch.Tensor, scale: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from a Cauchy
    distribution.

    Unlike ``rand_cauchy``, this function allows to sample values
    from different Cauchy distributions at the same time.
    The shape of the ``loc`` and ``scale`` tensors are used to infer
    the output size.

    Args:
        loc: The location/median of the Cauchy distribution.
            It must be a float tensor of shape ``(d0, d1, ..., dn)``.
        scale: The standard deviation of the Cauchy
            distribution. It must be a float tensor of shape
            ``(d0, d1, ..., dn)``.
        generator: An optional random generator.

    Returns:
        A tensor of shape ``(d0, d1, ..., dn)`` filled with values
            sampled from a Cauchy distribution.

    Raises:
        ValueError: if the ``loc`` and ``scale`` parameters are not
            valid.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.random import cauchy
    >>> cauchy(loc=torch.tensor([-1.0, 0.0, 1.0]), scale=torch.tensor([1.0, 3.0, 5.0]))
    tensor([...])

    ```
    """
    if loc.shape != scale.shape:
        msg = f"The shapes of loc and scale do not match ({loc.shape} vs {scale.shape})"
        raise ValueError(msg)
    if torch.any(scale.le(0.0)):
        msg = f"scale has to be greater than 0 (received: {scale})"
        raise ValueError(msg)
    return rand_cauchy(loc.shape, generator=generator).mul(scale).add(loc)


def rand_normal(
    size: list[int] | tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from a Normal
    distribution.

    Args:
        size: The tensor shape.
        mean: The mean of the Normal distribution.
        std: The standard deviation of the Normal
            distribution.
        generator: An optional random generator.

    Returns:
        A tensor filled with values sampled from a Normal distribution.

    Raises:
        ValueError: if the ``std`` parameter is not valid.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.random import rand_normal
    >>> rand_normal((2, 3), mean=1.0, std=2.0)
    tensor([[...]])

    ```
    """
    if std <= 0.0:
        msg = f"std has to be greater than 0 (received: {std})"
        raise ValueError(msg)
    return torch.randn(size, generator=generator).mul(std).add(mean)


def normal(
    mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from a Normal
    distribution.

    Args:
        mean: The mean. It must be a float tensor of shape
            ``(d0, d1, ..., dn)``.
        std: The standard deviation. It must be a float
            tensor of shape ``(d0, d1, ..., dn)``.
        generator: An optional random generator.

    Returns:
        A tensor of shape ``(d0, d1, ..., dn)`` filled with values
            sampled from a Normal distribution.

    Raises:
        ValueError: if the ``mean`` and ``std`` parameters are not
            valid.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.random import normal
    >>> normal(mean=torch.tensor([-1.0, 0.0, 1.0]), std=torch.tensor([1.0, 3.0, 5.0]))
    tensor([...])

    ```
    """
    if mean.shape != std.shape:
        msg = f"The shapes of mean and std do not match ({mean.shape} vs {std.shape})"
        raise ValueError(msg)
    if torch.any(std.le(0.0)):
        msg = f"std has to be greater than 0 (received: {std})"
        raise ValueError(msg)
    return torch.randn(mean.shape, generator=generator).mul(std).add(mean)
