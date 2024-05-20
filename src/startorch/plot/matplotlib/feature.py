r"""Contain functions to plot features distribution."""

from __future__ import annotations

__all__ = ["hist_feature"]

import math
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import torch

from startorch.utils.imports import check_matplotlib, is_matplotlib_available

if is_matplotlib_available():
    from matplotlib import pyplot as plt
else:  # pragma: no cover
    plt = Mock()


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


def hist_feature(
    features: torch.Tensor | np.ndarray,
    feature_names: Sequence[str] | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] = (6, 4),
    **kwargs: Any,
) -> plt.Figure:
    r"""Plot the distribution of each feature.

    If the input has ``n`` features, this function returns a figure
    with ``n`` histograms: one for each features.

    Args:
        features: Specifies the features. It must be a tensor of shape
            ``(d0, d1, ..., dn)``.
        feature_names: Specifies the feature names. If ``None``,
            the feature names are generated automatically.
        ncols: Specifies the number of columns.
        figsize: Specifies the individual figure size in pixels.
            The first dimension is the width and the second is the
            height.
        **kwargs: Additional keyword arguments for ``plt.hist``.

    Returns:
        ``matplotlib.pyplot.Figure``: The generated figure.

    Raises:
        RuntimeError: if the ``features`` shape is invalid
        RuntimeError: if ``features`` and ``feature_names`` are not
            consistent

    Example usage:

    ```pycon

    >>> from startorch.plot.matplotlib import hist_feature
    >>> import numpy as np
    >>> fig = hist_feature(np.random.rand(10, 5))

    ```
    """
    check_matplotlib()
    if torch.is_tensor(features):
        features = features.numpy()
    if features.ndim != 2:
        msg = f"Expected a 2D array/tensor but received {features.ndim} dimensions"
        raise RuntimeError(msg)
    feature_size = features.shape[1]
    if feature_names is None:
        feature_names = [f"feature {i}" for i in range(feature_size)]
    elif len(feature_names) != feature_size:
        msg = (
            f"The number of features ({feature_size:,}) does not match with the number of "
            f"feature names ({len(feature_names):,})"
        )
        raise RuntimeError(msg)

    nrows = math.ceil(feature_size / ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), squeeze=False
    )
    for i in range(feature_size):
        x, y = i // ncols, i % ncols
        axes[x, y].hist(features[:, i], **kwargs)
        axes[x, y].set_title(feature_names[i])
    return fig
