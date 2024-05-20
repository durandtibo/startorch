from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from startorch.sequence import RandInt
from startorch.transformer.tensor import (
    Identity,
    is_tensor_transformer_config,
    setup_tensor_transformer,
)

if TYPE_CHECKING:
    import pytest

##################################################
#     Tests for is_tensor_transformer_config     #
##################################################


def test_is_tensor_transformer_config_true() -> None:
    assert is_tensor_transformer_config({OBJECT_TARGET: "startorch.transformer.tensor.Identity"})


def test_is_tensor_transformer_config_false() -> None:
    assert not is_tensor_transformer_config({OBJECT_TARGET: "torch.nn.Identity"})


##############################################
#     Tests for setup_tensor_transformer     #
##############################################


def test_setup_tensor_transformer_object() -> None:
    transformer = Identity()
    assert setup_tensor_transformer(transformer) is transformer


def test_setup_tensor_transformer_dict() -> None:
    assert isinstance(
        setup_tensor_transformer({OBJECT_TARGET: "startorch.transformer.tensor.Identity"}),
        Identity,
    )


def test_setup_tensor_transformer_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_tensor_transformer(
                {OBJECT_TARGET: "startorch.sequence.RandInt", "low": 0, "high": 10}
            ),
            RandInt,
        )
        assert caplog.messages
