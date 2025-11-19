from __future__ import annotations

import pytest
import torch

from startorch import constants as ct
from startorch.example import CirclesClassification

SIZES = [1, 2, 4]


###########################################################
#     Tests for CirclesClassificationExampleGenerator     #
###########################################################


@pytest.mark.parametrize("batch_size", SIZES)
def test_circles_classification_generate(batch_size: int) -> None:
    data = CirclesClassification().generate(batch_size)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (batch_size,)
    assert data[ct.TARGET].dtype == torch.long
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (batch_size, 2)
    assert data[ct.FEATURE].dtype == torch.float
