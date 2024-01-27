r"""Contain functions to plot features distribution."""

from __future__ import annotations

__all__ = ["hist_feature"]

import math
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from startorch.utils.conversion import to_array
from startorch.utils.imports import check_plotly, is_plotly_available

if is_plotly_available():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
else:  # pragma: no cover
    go = Mock()


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import torch


def hist_feature(
    features: torch.Tensor | np.ndarray,
    feature_names: Sequence[str] | None = None,
    ncols: int = 2,
    figsize: tuple[int, int] = (250, 200),
    **kwargs: Any,
) -> go.Figure:
    r"""Plot the distribution of each feature.

    If the input has ``n`` features, this function returns a figure
    with ``n`` histograms: one for each feature.

    Args:
        features: Specifies the features. It must be a tensor of shape
            ``(d0, d1, ..., dn)``.
        feature_names: Specifies the feature names. If ``None``,
            the feature names are generated automatically.
        ncols: Specifies the number of columns.
        figsize: Specifies the individual figure size in pixels.
            The first dimension is the width and the second is the
            height.
        **kwargs: Additional keyword arguments for
            ``plotly.graph_objects.Histogram``.

    Returns:
        The generated figure.

    Raises:
        RuntimeError: if the ``features`` shape is invalid
        RuntimeError: if ``features`` and ``feature_names`` are not
            consistent

    Example usage:

    ```pycon
    >>> from startorch.plot.plotly import hist_feature
    >>> import numpy as np
    >>> fig = hist_feature(np.random.rand(10, 5))

    ```
    """
    check_plotly()
    features = to_array(features)
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
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=feature_names)
    for i in range(feature_size):
        x, y = i // ncols, i % ncols
        fig.add_trace(
            go.Histogram(x=features[:, i], **kwargs, name=feature_names[i]), row=x + 1, col=y + 1
        )

    fig.update_layout(height=figsize[1] * nrows + 120, width=figsize[0] * ncols, showlegend=False)
    return fig
