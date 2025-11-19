from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from startorch.timeseries.utils import mix2sequences

###################################
#     Tests for mix2sequences     #
###################################


def test_mix2sequences_seq_dim_1() -> None:
    assert objects_are_equal(
        mix2sequences(
            torch.tensor([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]]),
            torch.tensor([[10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5]]),
        ),
        (
            torch.tensor([[0, 11, 2, 13, 4, 15], [10, 1, 12, 3, 14, 5]]),
            torch.tensor([[10, 1, 12, 3, 14, 5], [0, 11, 2, 13, 4, 15]]),
        ),
    )


def test_mix2sequences_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match=r"x and y shapes do not match:"):
        mix2sequences(torch.ones(1, 5), torch.ones(1, 5, 1))
