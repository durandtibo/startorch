from __future__ import annotations

from unittest.mock import patch

import pytest
from coola import objects_are_equal

from startorch.plot.plotly import hist_sequence, plot_sequence
from startorch.sequence import BaseSequenceGenerator, RandNormal, RandUniform
from startorch.testing import plotly_available
from startorch.utils.imports import is_plotly_available
from startorch.utils.seed import get_torch_generator

if is_plotly_available():
    import plotly.graph_objects as go

###################################
#     Tests for hist_sequence     #
###################################


@plotly_available
@pytest.mark.parametrize("generator", [RandUniform(), RandNormal()])
def test_hist_sequence_generator(generator: BaseSequenceGenerator) -> None:
    assert isinstance(hist_sequence(generator, batch_size=2, seq_len=6), go.Figure)


@plotly_available
@pytest.mark.parametrize("scale", ["identity", "log", "log10", "log2", "log1p", "asinh"])
def test_hist_sequence_scale(scale: str) -> None:
    assert isinstance(hist_sequence(RandUniform(), batch_size=2, seq_len=6, scale=scale), go.Figure)


@patch("startorch.utils.imports.is_plotly_available", lambda: True)
@pytest.mark.parametrize("seq_len", [1, 2, 4])
def test_hist_sequence_seq_len(seq_len: int) -> None:
    with patch("startorch.plot.plotly.sequence.go") as go:
        hist_sequence(RandUniform(), seq_len=seq_len, batch_size=1)
        assert go.Histogram.call_args.kwargs["x"].shape[0] == seq_len


@patch("startorch.utils.imports.is_plotly_available", lambda: True)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_hist_sequence_batch_size(batch_size: int) -> None:
    with patch("startorch.plot.plotly.sequence.go") as go:
        hist_sequence(RandUniform(), seq_len=1, batch_size=batch_size)
        assert go.Histogram.call_args.kwargs["x"].shape[0] == batch_size


@patch("startorch.utils.imports.is_plotly_available", lambda: True)
@pytest.mark.parametrize("num_batches", [1, 2, 4])
def test_hist_sequence_num_batches(num_batches: int) -> None:
    with patch("startorch.plot.plotly.sequence.go") as go:
        hist_sequence(RandUniform(), seq_len=1, batch_size=1, num_batches=num_batches)
        assert go.Histogram.call_args.kwargs["x"].shape[0] == num_batches


@patch("startorch.utils.imports.is_plotly_available", lambda: True)
@pytest.mark.parametrize("seed", [1, 2, 4])
def test_hist_sequence_rng(seed: int) -> None:
    with patch("startorch.plot.plotly.sequence.go") as go:
        hist_sequence(RandUniform(), rng=seed)
        hist_sequence(RandUniform(), rng=get_torch_generator(seed))
        assert objects_are_equal(
            go.Histogram.call_args_list[0].kwargs["x"],
            go.Histogram.call_args_list[1].kwargs["x"],
        )


@patch("startorch.utils.imports.is_plotly_available", lambda: False)
def test_hist_sequence_no_plotly() -> None:
    with pytest.raises(RuntimeError, match="`plotly` package is required but not installed."):
        hist_sequence(RandUniform())


###################################
#     Tests for plot_sequence     #
###################################


@plotly_available
@pytest.mark.parametrize("generator", [RandUniform(), RandNormal()])
def test_plot_sequence_generator(generator: BaseSequenceGenerator) -> None:
    assert isinstance(plot_sequence(generator, batch_size=2, seq_len=6), go.Figure)


@patch("startorch.utils.imports.is_plotly_available", lambda: True)
@pytest.mark.parametrize("seq_len", [1, 2, 4])
def test_plot_sequence_seq_len(seq_len: int) -> None:
    with patch("startorch.plot.plotly.sequence.go") as go:
        plot_sequence(RandUniform(), seq_len=seq_len)
        assert go.Scatter.call_args.kwargs["x"].shape[0] == seq_len


@patch("startorch.utils.imports.is_plotly_available", lambda: True)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_plot_sequence_batch_size(batch_size: int) -> None:
    with patch("startorch.plot.plotly.sequence.go") as go:
        plot_sequence(RandUniform(), batch_size=batch_size)
        assert len(go.Scatter.call_args_list) == batch_size


@patch("startorch.utils.imports.is_plotly_available", lambda: True)
@pytest.mark.parametrize("num_batches", [1, 2, 4])
def test_plot_sequence_num_batches(num_batches: int) -> None:
    with patch("startorch.plot.plotly.sequence.go") as go:
        plot_sequence(RandUniform(), batch_size=1, num_batches=num_batches)
        assert len(go.Scatter.call_args_list) == num_batches


@patch("startorch.utils.imports.is_plotly_available", lambda: True)
@pytest.mark.parametrize("seed", [1, 2, 4])
def test_plot_sequence_rng(seed: int) -> None:
    with patch("startorch.plot.plotly.sequence.go") as go:
        plot_sequence(RandUniform(), rng=seed)
        plot_sequence(RandUniform(), rng=get_torch_generator(seed))
        assert objects_are_equal(
            go.Scatter.call_args_list[0].kwargs["x"],
            go.Scatter.call_args_list[1].kwargs["x"],
        )
        assert objects_are_equal(
            go.Scatter.call_args_list[0].kwargs["y"],
            go.Scatter.call_args_list[1].kwargs["y"],
        )


@patch("startorch.utils.imports.is_plotly_available", lambda: False)
def test_plot_sequence_no_plotly() -> None:
    with pytest.raises(RuntimeError, match="`plotly` package is required but not installed."):
        plot_sequence(RandUniform())
