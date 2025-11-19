from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from coola import objects_are_equal

from startorch.plot.matplotlib import hist_sequence, plot_sequence
from startorch.sequence import BaseSequenceGenerator, RandNormal, RandUniform
from startorch.testing import matplotlib_available
from startorch.utils.imports import is_matplotlib_available
from startorch.utils.seed import get_torch_generator

if is_matplotlib_available():
    from matplotlib import pyplot as plt

###################################
#     Tests for hist_sequence     #
###################################


@matplotlib_available
@pytest.mark.parametrize("generator", [RandUniform(), RandNormal()])
def test_hist_sequence_generator(generator: BaseSequenceGenerator) -> None:
    assert isinstance(hist_sequence(generator, batch_size=2, seq_len=6), plt.Figure)


@matplotlib_available
@pytest.mark.parametrize("scale", ["identity", "log", "log10", "log2", "log1p", "asinh"])
def test_hist_sequence_scale(scale: str) -> None:
    assert isinstance(
        hist_sequence(RandUniform(), batch_size=2, seq_len=6, scale=scale), plt.Figure
    )


@patch("startorch.utils.imports.is_matplotlib_available", lambda: True)
@pytest.mark.parametrize("seq_len", [1, 2, 4])
def test_hist_sequence_seq_len(seq_len: int) -> None:
    ax = Mock()
    with patch(
        "startorch.plot.matplotlib.sequence.plt.subplots",
        lambda *args, **kwargs: (Mock(), ax),  # noqa: ARG005
    ):
        hist_sequence(RandUniform(), seq_len=seq_len, batch_size=1)
        assert ax.hist.call_args_list[0].args[0].shape[0] == seq_len


@patch("startorch.utils.imports.is_matplotlib_available", lambda: True)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_hist_sequence_batch_size(batch_size: int) -> None:
    ax = Mock()
    with patch(
        "startorch.plot.matplotlib.sequence.plt.subplots",
        lambda *args, **kwargs: (Mock(), ax),  # noqa: ARG005
    ):
        hist_sequence(RandUniform(), seq_len=1, batch_size=batch_size)
        assert ax.hist.call_args_list[0].args[0].shape[0] == batch_size


@patch("startorch.utils.imports.is_matplotlib_available", lambda: True)
@pytest.mark.parametrize("num_batches", [1, 2, 4])
def test_hist_sequence_num_batches(num_batches: int) -> None:
    ax = Mock()
    with patch(
        "startorch.plot.matplotlib.sequence.plt.subplots",
        lambda *args, **kwargs: (Mock(), ax),  # noqa: ARG005
    ):
        hist_sequence(RandUniform(), seq_len=1, batch_size=1, num_batches=num_batches)
        assert ax.hist.call_args_list[0].args[0].shape[0] == num_batches


@patch("startorch.utils.imports.is_matplotlib_available", lambda: True)
@pytest.mark.parametrize("seed", [1, 2, 4])
def test_hist_sequence_rng(seed: int) -> None:
    ax = Mock()
    with patch(
        "startorch.plot.matplotlib.sequence.plt.subplots",
        lambda *args, **kwargs: (Mock(), ax),  # noqa: ARG005
    ):
        hist_sequence(RandUniform(), rng=seed)
        hist_sequence(RandUniform(), rng=get_torch_generator(seed))
        assert objects_are_equal(
            ax.hist.call_args_list[0].args[0], ax.hist.call_args_list[1].args[0]
        )


@patch("startorch.utils.imports.is_matplotlib_available", lambda: False)
def test_hist_sequence_no_matplotlib() -> None:
    with pytest.raises(RuntimeError, match=r"`matplotlib` package is required but not installed."):
        hist_sequence(RandUniform())


###################################
#     Tests for plot_sequence     #
###################################


@matplotlib_available
@pytest.mark.parametrize("generator", [RandUniform(), RandNormal()])
def test_plot_sequence_generator(generator: BaseSequenceGenerator) -> None:
    assert isinstance(plot_sequence(generator, batch_size=2, seq_len=6), plt.Figure)


@patch("startorch.utils.imports.is_matplotlib_available", lambda: True)
@pytest.mark.parametrize("seq_len", [1, 2, 4])
def test_plot_sequence_seq_len(seq_len: int) -> None:
    ax = Mock()
    with patch(
        "startorch.plot.matplotlib.sequence.plt.subplots",
        lambda *args, **kwargs: (Mock(), ax),  # noqa: ARG005
    ):
        plot_sequence(RandUniform(), seq_len=seq_len)
        assert ax.plot.call_args_list[0].args[0].shape[0] == seq_len


@patch("startorch.utils.imports.is_matplotlib_available", lambda: True)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_plot_sequence_batch_size(batch_size: int) -> None:
    ax = Mock()
    with patch(
        "startorch.plot.matplotlib.sequence.plt.subplots",
        lambda *args, **kwargs: (Mock(), ax),  # noqa: ARG005
    ):
        plot_sequence(RandUniform(), batch_size=batch_size)
        assert len(ax.plot.call_args_list) == batch_size


@patch("startorch.utils.imports.is_matplotlib_available", lambda: True)
@pytest.mark.parametrize("num_batches", [1, 2, 4])
def test_plot_sequence_num_batches(num_batches: int) -> None:
    ax = Mock()
    with patch(
        "startorch.plot.matplotlib.sequence.plt.subplots",
        lambda *args, **kwargs: (Mock(), ax),  # noqa: ARG005
    ):
        plot_sequence(RandUniform(), batch_size=1, num_batches=num_batches)
        assert len(ax.plot.call_args_list) == num_batches


@patch("startorch.utils.imports.is_matplotlib_available", lambda: True)
@pytest.mark.parametrize("seed", [1, 2, 4])
def test_plot_sequence_rng(seed: int) -> None:
    ax = Mock()
    with patch(
        "startorch.plot.matplotlib.sequence.plt.subplots",
        lambda *args, **kwargs: (Mock(), ax),  # noqa: ARG005
    ):
        plot_sequence(RandUniform(), rng=seed)
        plot_sequence(RandUniform(), rng=get_torch_generator(seed))
        assert objects_are_equal(
            ax.plot.call_args_list[0].args[0], ax.plot.call_args_list[1].args[0]
        )


@patch("startorch.utils.imports.is_matplotlib_available", lambda: False)
def test_plot_sequence_no_matplotlib() -> None:
    with pytest.raises(RuntimeError, match=r"`matplotlib` package is required but not installed."):
        plot_sequence(RandUniform())
