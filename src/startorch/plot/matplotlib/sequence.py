r"""Contain functions to plot the distribution of values in
sequences."""

from __future__ import annotations

__all__ = ["hist_sequence", "plot_sequence"]

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from batchtensor.constants import BATCH_DIM
from batchtensor.tensor import select_along_batch

from startorch.utils.batch import merge_batches, scale_batch
from startorch.utils.imports import check_matplotlib, is_matplotlib_available
from startorch.utils.seed import setup_torch_generator

if is_matplotlib_available():
    from matplotlib import pyplot as plt
else:  # pragma: no cover
    plt = Mock()

if TYPE_CHECKING:
    from torch import Generator

    from startorch.sequence.base import BaseSequenceGenerator


def hist_sequence(
    sequence: BaseSequenceGenerator,
    bins: int = 500,
    seq_len: int = 1000,
    batch_size: int = 10000,
    num_batches: int = 1,
    rng: int | Generator = 13683624337160779813,
    figsize: tuple[float, float] = (16, 5),
    scale: str = "identity",
    **kwargs: Any,
) -> plt.Figure:
    r"""Plot the distribution from a sequence generator.

    Args:
        sequence: Specifies the sequence generator.
        bins: Specifies the number of histogram bins.
        seq_len: Specifies the sequence length.
        batch_size: Specifies the batch size.
        num_batches: Specifies the number of batches to generate.
        rng: Specifies a random number generator or a random seed.
        figsize: Specifies the figure size.
        scale: Specifies a scale transformation of the features.
        **kwargs: Additional keyword arguments for ``plt.hist``.

    Returns:
        The generated figure.

    Example usage:

    ```pycon
    >>> from startorch.plot.matplotlib import hist_sequence
    >>> from startorch.sequence import RandUniform
    >>> fig = hist_sequence(RandUniform(low=-5, high=5))

    ```
    """
    check_matplotlib()
    rng = setup_torch_generator(rng)

    batch = merge_batches(
        [
            sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
            for _ in range(num_batches)
        ]
    )
    batch = scale_batch(batch, scale=scale)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(batch.flatten().numpy(), bins=bins, **kwargs)
    return fig


def plot_sequence(
    sequence: BaseSequenceGenerator,
    seq_len: int = 128,
    batch_size: int = 1,
    num_batches: int = 1,
    rng: int | Generator = 13683624337160779813,
    figsize: tuple[float, float] = (16, 5),
    xscale: str = "linear",
    yscale: str = "linear",
    **kwargs: Any,
) -> plt.Figure:
    r"""Plot some sequences generated from a sequence generator.

    Args:
        sequence: Specifies the sequence generator.
        seq_len: Specifies the sequence length.
        batch_size: Specifies the batch size.
        num_batches: Specifies the number of batches.
        rng: Specifies a random number generator or a random seed.
        figsize: Specifies the figure size.
        xscale: Specifies the x-axis scale.
        yscale: Specifies the y-axis scale.
        **kwargs: Additional keyword arguments for ``plt.plot``.

    Returns:
        The generated figure.

    Example usage:

    ```pycon
    >>> from startorch.plot.matplotlib import plot_sequence
    >>> from startorch.sequence import RandUniform
    >>> fig = plot_sequence(RandUniform(low=-5, high=5), batch_size=4)

    ```
    """
    check_matplotlib()
    rng = setup_torch_generator(rng)

    fig, ax = plt.subplots(figsize=figsize)
    for _ in range(num_batches):
        batch = sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        for i in range(batch.shape[BATCH_DIM]):
            ax.plot(select_along_batch(batch, i).flatten().numpy(), marker="o", **kwargs)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    return fig
