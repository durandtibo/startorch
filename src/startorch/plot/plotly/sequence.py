from __future__ import annotations

__all__ = ["hist_sequence", "plot_sequence"]

from unittest.mock import Mock

import numpy as np
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator
from startorch.utils.batch import merge_batches, scale_batch
from startorch.utils.imports import check_plotly, is_plotly_available
from startorch.utils.seed import get_torch_generator

if is_plotly_available():
    import plotly.graph_objects as go
else:
    go = Mock()  # pragma: no cover


def hist_sequence(
    sequence: BaseSequenceGenerator,
    bins: int = 500,
    seq_len: int = 1000,
    batch_size: int = 10000,
    num_batches: int = 1,
    rng: int | Generator = 13683624337160779813,
    figsize: tuple[int, int] = (800, 600),
    scale: str = "identity",
    **kwargs,
) -> go.Figure:
    r"""Plots the distribution from a sequence generator.

    Args:
        sequence (``BaseSequenceGenerator``): Specifies the sequence
            generator.
        bins: Specifies the number of histogram bins.
            Default: ``500``
        seq_len: Specifies the sequence length.
            Default: ``128``
        batch_size: Specifies the batch size.
            Default: ``1``
        rng (``torch.Generator`` or int): Specifies a random number
            generator or a random seed.
            Default: ``13683624337160779813``
        figsize (tuple, optional): Specifies the figure size.
            Default: ``(800, 600)``
        **kwargs: Additional keyword arguments for
            ``plotly.graph_objects.Histogram``.

    Returns:
        ``plotly.graph_objects.Figure``: The generated figure.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.plot.plotly import hist_sequence
        >>> from startorch.sequence import RandUniform
        >>> fig = hist_sequence(RandUniform(low=-5, high=5))
    """
    check_plotly()
    if not isinstance(rng, Generator):
        rng = get_torch_generator(random_seed=rng)

    batch = merge_batches(
        [
            sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
            for _ in range(num_batches)
        ]
    )
    batch = scale_batch(batch, scale=scale)
    fig = go.Figure(data=[go.Histogram(x=batch.data.flatten().numpy(), nbinsx=bins, **kwargs)])
    fig.update_layout(height=figsize[1], width=figsize[0], showlegend=False)
    return fig


def plot_sequence(
    sequence: BaseSequenceGenerator,
    seq_len: int = 128,
    batch_size: int = 1,
    num_batches: int = 1,
    rng: int | Generator = 13683624337160779813,
    **kwargs,
) -> go.Figure:
    r"""Plots some sequences generated from a sequence generator.

    Args:
        sequence (``BaseSequenceGenerator``): Specifies the sequence
            generator.
        seq_len: Specifies the sequence length.
            Default: ``128``
        batch_size: Specifies the batch size.
            Default: ``1``
        num_batches: Specifies the number of batches.
            Default: ``1``
        rng (``torch.Generator`` or int): Specifies a random number
            generator or a random seed.
            Default: ``13683624337160779813``
        figsize (tuple, optional): Specifies the figure size.
            Default: ``(16, 5)``
        **kwargs: Additional keyword arguments for ``plt.plot``.

    Returns:
        ``plotly.graph_objects.Figure``: The generated figure.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.plot.plotly import plot_sequence
        >>> from startorch.sequence import RandUniform
        >>> fig = plot_sequence(RandUniform(low=-5, high=5), batch_size=4)
    """
    check_plotly()
    if not isinstance(rng, Generator):
        rng = get_torch_generator(random_seed=rng)

    fig = go.Figure()
    for _ in range(num_batches):
        batch = sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        for i in range(batch.batch_size):
            y = batch.select_along_batch(i).data.flatten().numpy()
            fig.add_trace(go.Scatter(x=np.arange(y.shape[0]), y=y, **kwargs))
    return fig
