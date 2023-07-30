from __future__ import annotations

from unittest.mock import Mock, patch

from coola import objects_are_equal
from pytest import mark

from startorch.sequence import BaseSequenceGenerator, RandNormal, RandUniform
from startorch.testing import matplotlib_available
from startorch.utils.imports import is_matplotlib_available
from startorch.utils.plot import plot_sequence
from startorch.utils.seed import get_torch_generator

if is_matplotlib_available():
    from matplotlib import pyplot as plt


###################################
#     Tests for plot_sequence     #
###################################


@matplotlib_available
@mark.parametrize("generator", (RandUniform(), RandNormal()))
def test_plot_sequence_generator(generator: BaseSequenceGenerator) -> None:
    assert isinstance(plot_sequence(generator), plt.Figure)


@mark.parametrize("seq_len", (1, 2, 4))
def test_plot_sequence_seq_len(seq_len: int) -> None:
    ax = Mock()
    with patch("startorch.utils.plot.plt.subplots", lambda *args, **kwargs: (Mock(), ax)):
        plot_sequence(RandUniform(), seq_len=seq_len)
        assert ax.plot.call_args_list[0].args[0].shape[0] == seq_len


@mark.parametrize("batch_size", (1, 2, 4))
def test_plot_sequence_batch_size(batch_size: int) -> None:
    ax = Mock()
    with patch("startorch.utils.plot.plt.subplots", lambda *args, **kwargs: (Mock(), ax)):
        plot_sequence(RandUniform(), batch_size=batch_size)
        assert len(ax.plot.call_args_list) == batch_size


@mark.parametrize("seed", (1, 2, 4))
def test_plot_sequence_rng(seed: int) -> None:
    ax = Mock()
    with patch("startorch.utils.plot.plt.subplots", lambda *args, **kwargs: (Mock(), ax)):
        plot_sequence(RandUniform(), rng=seed)
        plot_sequence(RandUniform(), rng=get_torch_generator(seed))
        assert objects_are_equal(
            ax.plot.call_args_list[0].args[0], ax.plot.call_args_list[1].args[0]
        )
