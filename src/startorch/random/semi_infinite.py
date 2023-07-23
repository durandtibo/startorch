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
from torch import Generator, Tensor

from startorch.random import cauchy, normal, rand_cauchy, rand_normal


def rand_exponential(
    size: list[int] | tuple[int, ...],
    rate: float = 1.0,
    generator: Generator | None = None,
) -> Tensor:
    r"""Creates a tensor filled with values sampled from an Exponential
    distribution.

    Args:
    ----
        size (list or tuple): Specifies the tensor shape.
        rate (float, optional): Specifies the rate of the Exponential
            distribution. Default: ``1.0``
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from an
            Exponential distribution

    Raises:
    ------
        ValueError if the ``rate`` parameter is not valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import rand_exponential
        >>> rand_exponential((2, 3), rate=1.0)  # doctest:+ELLIPSIS
        tensor([[...]])
    """
    if rate <= 0:
        raise ValueError(f"rate has to be greater than 0 (received: {rate})")
    tensor = torch.zeros(*size, dtype=torch.float)
    tensor.exponential_(rate, generator=generator)
    return tensor


def exponential(rate: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    r"""Creates a tensor filled with values sampled from an Exponential
    distribution.

    Unlike ``rand_exponential``, this function allows to sample values
    from different Exponential distributions at the same time.
    The shape of the ``rate`` tensor is used to infer the output size.

    Args:
    ----
        rate (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the rates
            of the Exponential distribution. The output has the
            same shape that this input.
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``: A tensor filled with values sampled
            from an Exponential distribution.

    Raises:
    ------
        ValueError if the ``rate`` parameter is not valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import exponential
        >>> exponential(torch.tensor([1.0, 3.0, 5.0]))  # doctest:+ELLIPSIS
        tensor([...])
    """
    rate = rate.float()
    if torch.any(rate <= 0.0):
        raise ValueError("rate values have to be greater than 0 (>0)")
    return torch.zeros_like(rate).exponential_(generator=generator) / rate


def rand_half_cauchy(
    size: list[int] | tuple[int, ...],
    scale: float = 1.0,
    generator: Generator | None = None,
) -> Tensor:
    r"""Creates a tensor filled with values sampled from a half-Cauchy
    distribution.

    Args:
    ----
        size (list or tuple): Specifies the tensor shape.
        scale (float, optional): Specifies the scale of the
            half-Cauchy distribution. This value has to be greater
            than 0. Default: ``1.0``
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from a
            half-Cauchy distribution.

    Raises:
    ------
        ValueError if the ``scale`` parameter is not valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import rand_half_cauchy
        >>> rand_half_cauchy((2, 3), scale=1.0)  # doctest:+ELLIPSIS
        tensor([[...]])
    """
    return rand_cauchy(size=size, loc=0.0, scale=scale, generator=generator).abs()


def half_cauchy(scale: Tensor, generator: Generator | None = None) -> Tensor:
    r"""Creates a tensor filled with values sampled from a half-Cauchy
    distribution.

    Unlike ``rand_half_cauchy``, this function allows to sample values
    from different half-Cauchy distributions at the same time.
    The shape of the ``scale`` tensor is used to infer
    the output size.

    Args:
    ----
        scale (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the scale
            of the half-Cauchy distribution.
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from a
            half-Cauchy distribution.

    Raises:
    ------
        ValueError if the ``scale`` parameter is not valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import half_cauchy
        >>> half_cauchy(torch.tensor([1.0, 3.0, 5.0]))  # doctest:+ELLIPSIS
        tensor([...])
    """
    return cauchy(loc=torch.zeros_like(scale), scale=scale, generator=generator).abs()


def rand_half_normal(
    size: list[int] | tuple[int, ...],
    std: float = 1.0,
    generator: Generator | None = None,
) -> Tensor:
    r"""Creates a tensor filled with values sampled from a half-Normal
    distribution.

    Args:
    ----
        size (list or tuple): Specifies the tensor shape.
        std (float, optional): Specifies the standard deviation of
            the half-Normal distribution. Default: ``1.0``
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from a
            half-Normal distribution.

    Raises:
    ------
        ValueError if the ``std`` parameter is not valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import rand_half_normal
        >>> rand_half_normal((2, 3), std=1.0)  # doctest:+ELLIPSIS
        tensor([[...]])
    """
    if std <= 0:
        raise ValueError(f"std has to be greater than 0 (received: {std})")
    return rand_normal(size=size, mean=0.0, std=std, generator=generator).abs()


def half_normal(std: Tensor, generator: Generator | None = None) -> Tensor:
    r"""Creates a tensor filled with values sampled from a half-Normal
    distribution.

    Unlike ``rand_half_normal``, this function allows to sample values
    from different half-Normal distributions at the same time.
    The shape of the ``std`` tensor is used to infer
    the output size.

    Args:
    ----
        std (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the standard
            deviation of the half-Normal distribution.
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from a
            half-Normal distribution.

    Raises:
    ------
        ValueError if the ``std`` parameter is not valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import half_normal
        >>> half_normal(torch.tensor([1.0, 3.0, 5.0]))  # doctest:+ELLIPSIS
        tensor([...])
    """
    return normal(mean=torch.zeros_like(std), std=std, generator=generator).abs()


def rand_log_normal(
    size: list[int] | tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    generator: Generator | None = None,
) -> Tensor:
    r"""Creates a tensor filled with values sampled from a log-Normal
    distribution.

    Args:
    ----
        size (list or tuple): Specifies the tensor shape.
        mean (float, optional): Specifies the mean of the underlying
            Normal distribution. Default: ``0.0``
        std (float, optional): Specifies the standard deviation of
            the underlying Normal distribution. Default: ``1.0``
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from a
            log-Normal distribution.

    Raises:
    ------
        ValueError if the ``mean`` and ``std`` parametera are not
            valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import rand_log_normal
        >>> rand_log_normal((2, 3), mean=1.0, std=2.0)  # doctest:+ELLIPSIS
        tensor([[...]])
    """
    if std <= 0:
        raise ValueError(f"std has to be greater than 0 (received: {std})")
    tensor = torch.zeros(*size, dtype=torch.float)
    tensor.log_normal_(mean=mean, std=std, generator=generator)
    return tensor


def log_normal(mean: Tensor, std: Tensor, generator: Generator | None = None) -> Tensor:
    r"""Creates a tensor filled with values sampled from a log-Normal
    distribution.

    Unlike ``rand_log_normal``, this function allows to sample values
    from different log-Normal distributions at the same time.
    The shape of the ``mean`` and ``std`` tensors are used to infer
    the output size.

    Args:
    ----
        mean (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the mean
            of the log-Normal distribution.
        std (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the standard
            deviation of the log-Normal distribution.
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from a
            log-Normal distribution.

    Raises:
    ------
        ValueError if the ``mean`` and ``std`` parametera are not
            valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import log_normal
        >>> log_normal(
        ...     torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([1.0, 3.0, 5.0])
        ... )  # doctest:+ELLIPSIS
        tensor([...])
    """
    return normal(mean=mean, std=std, generator=generator).exp()
