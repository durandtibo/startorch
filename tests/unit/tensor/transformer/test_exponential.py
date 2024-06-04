from __future__ import annotations

from unittest.mock import Mock, patch

import torch
from coola import objects_are_equal

from startorch.tensor.transformer import ExponentialTensorTransformer
from startorch.utils.seed import get_torch_generator

##################################################
#     Tests for ExponentialTensorTransformer     #
##################################################


def test_exponential_str() -> None:
    assert str(ExponentialTensorTransformer()).startswith("ExponentialTensorTransformer(")


def test_exponential_transform() -> None:
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = ExponentialTensorTransformer().transform(tensor)
    assert tensor is not out
    assert tensor.shape == (2, 3)
    assert tensor.dtype == torch.float
    assert tensor.min() >= 0.0


def test_exponential_transform_mock() -> None:
    rate = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    transformer = ExponentialTensorTransformer()
    mock = Mock(return_value=torch.ones(2, 4))
    with patch("startorch.tensor.transformer.exponential.exponential", mock):
        assert transformer.transform(rate).equal(torch.ones(2, 4))
        mock.assert_called_once()


def test_exponential_transform_different_random_seeds() -> None:
    transformer = ExponentialTensorTransformer()
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert not objects_are_equal(
        transformer.transform(tensor, rng=get_torch_generator(1)),
        transformer.transform(tensor, rng=get_torch_generator(2)),
    )
