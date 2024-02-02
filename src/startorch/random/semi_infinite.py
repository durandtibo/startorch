r"""Contain functions to sample values from continuous uni-variate
distributions with semi-infinite support."""

from __future__ import annotations

__all__ = [
    "exponential",
    "half_cauchy",
    "half_normal",
    "log_normal",
    "rand_exponential",
    "rand_half_cauchy",
    "rand_half_normal",
    "rand_log_normal",
]


import torch

from startorch.random import cauchy, normal, rand_cauchy, rand_normal


def rand_exponential(
    size: list[int] | tuple[int, ...],
    rate: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from an Exponential
    distribution.

    Args:
        size: Specifies the tensor shape.
        rate: Specifies the rate of the Exponential distribution.
        generator: Specifies an optional random generator.

    Returns:
        A tensor filled with values sampled from an Exponential
            distribution.

    Raises:
        ValueError: if the ``rate`` parameter is not valid.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.random import rand_exponential
    >>> rand_exponential((2, 3), rate=1.0)
    tensor([[...]])

    ```
    """
    if rate <= 0:
        msg = f"rate has to be greater than 0 (received: {rate})"
        raise ValueError(msg)
    tensor = torch.zeros(*size, dtype=torch.float)
    tensor.exponential_(rate, generator=generator)
    return tensor


def exponential(rate: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from an Exponential
    distribution.

    Unlike ``rand_exponential``, this function allows to sample values
    from different Exponential distributions at the same time.
    The shape of the ``rate`` tensor is used to infer the output size.

    Args:
        rate: Specifies the rates of the Exponential distribution.
            It must be a float tensor of shape ``(d0, d1, ..., dn)``.
        generator: Specifies an optional random generator.

    Returns:
        A tensor of shape ``(d0, d1, ..., dn)`` filled with values
            sampled from an Exponential distribution.

    Raises:
        ValueError: if the ``rate`` parameter is not valid.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.random import exponential
    >>> exponential(torch.tensor([1.0, 3.0, 5.0]))
    tensor([...])

    ```
    """
    rate = rate.float()
    if torch.any(rate <= 0.0):
        msg = "rate values have to be greater than 0 (>0)"
        raise ValueError(msg)
    return torch.zeros_like(rate).exponential_(generator=generator) / rate


def rand_half_cauchy(
    size: list[int] | tuple[int, ...],
    scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from a half-Cauchy
    distribution.

    Args:
        size: Specifies the tensor shape.
        scale: Specifies the scale of the half-Cauchy distribution.
            This value has to be greater than 0.
        generator: Specifies an optional random generator.

    Returns:
        A tensor filled with values sampled from a half-Cauchy
            distribution.

    Raises:
        ValueError: if the ``scale`` parameter is not valid.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.random import rand_half_cauchy
    >>> rand_half_cauchy((2, 3), scale=1.0)
    tensor([[...]])

    ```
    """
    return rand_cauchy(size=size, loc=0.0, scale=scale, generator=generator).abs()


def half_cauchy(scale: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from a half-Cauchy
    distribution.

    Unlike ``rand_half_cauchy``, this function allows to sample values
    from different half-Cauchy distributions at the same time.
    The shape of the ``scale`` tensor is used to infer
    the output size.

    Args:
        scale: Specifies the scale of the half-Cauchy distribution.
            It must be a float tensor of shape ``(d0, d1, ..., dn)``.
        generator: Specifies an optional random generator.

    Returns:
        A tensor of shape ``(d0, d1, ..., dn)`` filled with values
            sampled from a half-Cauchy distribution.

    Raises:
        ValueError if the ``scale`` parameter is not valid.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.random import half_cauchy
    >>> half_cauchy(torch.tensor([1.0, 3.0, 5.0]))
    tensor([...])

    ```
    """
    return cauchy(loc=torch.zeros_like(scale), scale=scale, generator=generator).abs()


def rand_half_normal(
    size: list[int] | tuple[int, ...],
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from a half-Normal
    distribution.

    Args:
        size: Specifies the tensor shape.
        std: Specifies the standard deviation of the half-Normal
            distribution.
        generator: Specifies an optional random generator.

    Returns:
        A tensor filled with values sampled from a half-Normal
            distribution.

    Raises:
        ValueError: if the ``std`` parameter is not valid.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.random import rand_half_normal
    >>> rand_half_normal((2, 3), std=1.0)
    tensor([[...]])

    ```
    """
    if std <= 0:
        msg = f"std has to be greater than 0 (received: {std})"
        raise ValueError(msg)
    return rand_normal(size=size, mean=0.0, std=std, generator=generator).abs()


def half_normal(std: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from a half-Normal
    distribution.

    Unlike ``rand_half_normal``, this function allows to sample values
    from different half-Normal distributions at the same time.
    The shape of the ``std`` tensor is used to infer
    the output size.

    Args:
        std: Specifies the standard deviation of the half-Normal
            distribution. It must be a float tensor of shape
            ``(d0, d1, ..., dn)``.
        generator: Specifies
            an optional random generator.

    Returns:
        A tensor of shape ``(d0, d1, ..., dn)`` filled with values
            sampled from a half-Normal distribution.

    Raises:
        ValueError: if the ``std`` parameter is not valid.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.random import half_normal
    >>> half_normal(torch.tensor([1.0, 3.0, 5.0]))
    tensor([...])

    ```
    """
    return normal(mean=torch.zeros_like(std), std=std, generator=generator).abs()


def rand_log_normal(
    size: list[int] | tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from a log-Normal
    distribution.

    Args:
        size: Specifies the tensor shape.
        mean: Specifies the mean of the underlying Normal distribution.
        std: Specifies the standard deviation of the underlying
            Normal distribution.
        generator: Specifies an optional random generator.

    Returns:
        A tensor of shape ``(d0, d1, ..., dn)`` filled with values
            sampled from a log-Normal distribution.

    Raises:
        ValueError: if the ``mean`` and ``std`` parametera are not
            valid.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.random import rand_log_normal
    >>> rand_log_normal((2, 3), mean=1.0, std=2.0)
    tensor([[...]])

    ```
    """
    if std <= 0:
        msg = f"std has to be greater than 0 (received: {std})"
        raise ValueError(msg)
    tensor = torch.zeros(*size, dtype=torch.float)
    tensor.log_normal_(mean=mean, std=std, generator=generator)
    return tensor


def log_normal(
    mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    r"""Create a tensor filled with values sampled from a log-Normal
    distribution.

    Unlike ``rand_log_normal``, this function allows to sample values
    from different log-Normal distributions at the same time.
    The shape of the ``mean`` and ``std`` tensors are used to infer
    the output size.

    Args:
        mean: Specifies the mean of the log-Normal distribution.
            It must be a float tensor of shape ``(d0, d1, ..., dn)``.
        std: Specifies the standard deviation of the log-Normal
            distribution. It must be a float tensor of shape
            ``(d0, d1, ..., dn)``.
        generator: Specifies an optional random generator.

    Returns:
        A tensor of shape ``(d0, d1, ..., dn)`` filled with values
            sampled from a log-Normal distribution.

    Raises:
        ValueError: if the ``mean`` and ``std`` parametera are not
            valid.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.random import log_normal
    >>> log_normal(torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([1.0, 3.0, 5.0]))
    tensor([...])

    ```
    """
    return normal(mean=mean, std=std, generator=generator).exp()
