from __future__ import annotations

__all__ = ["hist_sequence", "plot_sequence"]

from unittest.mock import Mock

from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator
from startorch.utils.imports import check_matplotlib, is_matplotlib_available
from startorch.utils.seed import get_torch_generator

if is_matplotlib_available():
    from matplotlib import pyplot as plt
else:
    plt = Mock()  # pragma: no cover


def hist_sequence(
    sequence: BaseSequenceGenerator,
    bins: int = 500,
    seq_len: int = 1000,
    batch_size: int = 10000,
    rng: int | Generator = 13683624337160779813,
    figsize: tuple[int, int] = (16, 5),
) -> plt.Figure:
    r"""Plots the distribution from a sequence generator.

    Args:
    ----
        sequence (``BaseSequenceGenerator``): Specifies the sequence
            generator.
        bins (int, optional): Specifies the number of histogram bins.
            Default: ``500``
        seq_len (int, optional): Specifies the sequence length.
            Default: ``128``
        batch_size (int, optional): Specifies the batch size.
            Default: ``1``
        rng (``torch.Generator`` or int): Specifies a random number
            generator or a random seed.
            Default: ```13683624337160779813``
        figsize (tuple, optional): Specifies the figure size.
            Default: ``(16, 5)``

    Returns:
    -------
        ``matplotlib.pyplot.Figure``: The generated figure.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.utils.plot import hist_sequence
        >>> from startorch.sequence import RandUniform
        >>> generator = RandUniform(low=-5, high=5)
        >>> fig = hist_sequence(generator)
    """
    check_matplotlib()
    if not isinstance(rng, Generator):
        rng = get_torch_generator(random_seed=rng)

    batch = sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(batch.data.flatten().numpy(), bins=bins)
    return fig


def plot_sequence(
    sequence: BaseSequenceGenerator,
    seq_len: int = 128,
    batch_size: int = 1,
    rng: int | Generator = 13683624337160779813,
    figsize: tuple[int, int] = (16, 5),
) -> plt.Figure:
    r"""Plots some sequences generated from a sequence generator.

    Args:
    ----
        sequence (``BaseSequenceGenerator``): Specifies the sequence
            generator.
        seq_len (int, optional): Specifies the sequence length.
            Default: ``128``
        batch_size (int, optional): Specifies the batch size.
            Default: ``1``
        rng (``torch.Generator`` or int): Specifies a random number
            generator or a random seed.
            Default: ```13683624337160779813``
        figsize (tuple, optional): Specifies the figure size.
            Default: ``(16, 5)``

    Returns:
    -------
        ``matplotlib.pyplot.Figure``: The generated figure.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.utils.plot import plot_sequence
        >>> from startorch.sequence import RandUniform
        >>> generator = RandUniform(low=-5, high=5)
        >>> fig = plot_sequence(generator, batch_size=4)
    """
    check_matplotlib()
    if not isinstance(rng, Generator):
        rng = get_torch_generator(random_seed=rng)

    fig, ax = plt.subplots(figsize=figsize)
    batch = sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
    for i in range(batch.batch_size):
        ax.plot(batch.select_along_batch(i).data.flatten().numpy(), marker="o")
    return fig