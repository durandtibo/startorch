from __future__ import annotations

__all__ = ["cauchy", "normal", "rand_cauchy", "rand_normal"]


import torch
from torch import Tensor


def rand_cauchy(
    size: list[int] | tuple[int, ...],
    loc: float = 0.0,
    scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Creates a sequence of continuous variables sampled from a Cauchy
    distribution.

    Args:
    ----
        size (list or tuple): Specifies the tensor shape.
        loc (float, optional): Specifies the location/median of the
            Cauchy distribution. Default: ``0.0``
        scale (float, optional): Specifies the scale of the Cauchy
            distribution. This value has to be greater than 0.
            Default: ``1.0``
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from a
            Cauchy distribution.

    Raises:
    ------
        ValueError if the ``scale`` parameter is not valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import rand_cauchy
        >>> rand_cauchy((2, 3), loc=1.0, scale=2.0)  # doctest:+ELLIPSIS
        tensor([[...]])
    """
    if scale <= 0:
        raise ValueError(f"scale has to be greater than 0 (received: {scale})")
    sequence = torch.zeros(*size, dtype=torch.float)
    sequence.cauchy_(median=loc, sigma=scale, generator=generator)
    return sequence


def cauchy(loc: Tensor, scale: Tensor, generator: torch.Generator | None = None) -> Tensor:
    r"""Creates a tensor filled with values sampled from a Cauchy
    distribution.

    Unlike ``rand_cauchy``, this function allows to sample values
    from different Cauchy distributions at the same time.
    The shape of the ``loc`` and ``scale`` tensors are used to infer
    the output size.

    Args:
    ----
        loc (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the location/median
            of the Cauchy distribution.
        scale (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the standard
            deviation of the Cauchy distribution.
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from a
            Cauchy distribution.

    Raises:
    ------
        ValueError if the ``loc`` and ``scale`` parameters are not
            valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import cauchy
        >>> cauchy(
        ...     loc=torch.tensor([-1.0, 0.0, 1.0]), scale=torch.tensor([1.0, 3.0, 5.0])
        ... )  # doctest:+ELLIPSIS
        tensor([...])
    """
    if loc.shape != scale.shape:
        raise ValueError(f"The shapes of loc and scale do not match ({loc.shape} vs {scale.shape})")
    if torch.any(scale <= 0.0):
        raise ValueError(f"scale has to be greater than 0 (received: {scale})")
    return rand_cauchy(loc.shape, generator=generator).mul(scale).add(loc)


def rand_normal(
    size: list[int] | tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Creates a tensor filled with values sampled from a Normal
    distribution.

    Args:
    ----
        size (list or tuple): Specifies the tensor shape.
        mean (float, optional): Specifies the mean of the Normal
            distribution. Default: ``0.0``
        std (float, optional): Specifies the standard deviation of
            the Normal distribution. Default: ``1.0``
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from a
            Normal distribution.

    Raises:
    ------
        ValueError if the ``std`` parameter is not valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import rand_normal
        >>> rand_normal((2, 3), mean=1.0, std=2.0)  # doctest:+ELLIPSIS
        tensor([[...]])
    """
    if std <= 0.0:
        raise ValueError(f"std has to be greater than 0 (received: {std})")
    return torch.randn(size, generator=generator).mul(std).add(mean)


def normal(mean: Tensor, std: Tensor, generator: torch.Generator | None = None) -> Tensor:
    r"""Creates a tensor filled with values sampled from a Normal
    distribution.

    Args:
    ----
        mean (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the mean.
        std (``torch.Tensor`` of type float and shape
            ``(d0, d1, ..., dn)``): Specifies the standard
            deviation.
        generator (``torch.Generator`` or None, optional): Specifies
            an optional random generator. Default: ``None``

    Returns:
    -------
        ``torch.Tensor``: A tensor filled with values sampled from a
            Normal distribution.

    Raises:
    ------
        ValueError if the ``mean`` and ``std`` parameters are not
            valid.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.random import normal
        >>> normal(
        ...     mean=torch.tensor([-1.0, 0.0, 1.0]), std=torch.tensor([1.0, 3.0, 5.0])
        ... )  # doctest:+ELLIPSIS
        tensor([...])
    """
    if mean.shape != std.shape:
        raise ValueError(f"The shapes of mean and std do not match ({mean.shape} vs {std.shape})")
    if torch.any(std <= 0.0):
        raise ValueError(f"std has to be greater than 0 (received: {std})")
    return torch.randn(mean.shape, generator=generator).mul(std).add(mean)
