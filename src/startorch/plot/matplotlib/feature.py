from __future__ import annotations

__all__ = ["plot_features"]

import math
from collections.abc import Sequence
from unittest.mock import Mock

import numpy as np
import torch

from startorch.utils.imports import check_matplotlib, is_matplotlib_available

if is_matplotlib_available():
    from matplotlib import pyplot as plt
else:
    plt = Mock()  # pragma: no cover


def plot_features(
    features: torch.Tensor | np.ndarray,
    feature_names: Sequence[str] | None = None,
    ncols: int = 2,
    figsize: tuple[int, int] = (6, 4),
    **kwargs,
) -> plt.Figure:
    r"""

    Args:
    ----
        features (``torch.Tensor`` or ``numpy.ndarray`` of shape
            ``(batch_size, feature_size)``): Specifies the features.
        feature_names (``Sequence`` or ``None``, optional): Specifies
            the feature names. If ``None``, the feature names are
            generated automatically. Default: ``None``
        ncols (int, optional): Specifies the number of columns.
            Default: ``2``
        figsize (``tuple``, optional): Specifies the individual figure
            size. Default: ``(6, 4)``
        **kwargs: Additional keyword arguments for ``plt.hist``.

    Returns:
    -------
        ``matplotlib.pyplot.Figure``: The generated figure.

    Raises:
    ------
        RuntimeError if the ``features`` shape is invalid
        RuntimeError if ``features`` and ``feature_names`` are not
            consistent

    Example usage:

    .. code-block:: pycon

        >>> from startorch.plot.matplotlib import plot_features
        >>> import numpy as np
        >>> fig = plot_features(np.random.rand(10, 5))
    """
    check_matplotlib()
    if torch.is_tensor(features):
        features = features.numpy()
    if features.ndim != 2:
        raise RuntimeError(f"Expected a 2D array/tensor but received {features.ndim} dimensions")
    feature_size = features.shape[1]
    if feature_names is None:
        feature_names = [f"feature {i}" for i in range(feature_size)]
    elif len(feature_names) != feature_size:
        raise RuntimeError(
            f"The number of features ({feature_size:,}) does not match with the number of "
            f"feature names ({len(feature_names):,})"
        )

    nrows = math.ceil(feature_size / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), squeeze=False
    )
    for i in range(feature_size):
        x, y = i // ncols, i % ncols
        axes[x, y].hist(features[:, i], **kwargs)
        axes[x, y].set_title(feature_names[i])
    return fig
