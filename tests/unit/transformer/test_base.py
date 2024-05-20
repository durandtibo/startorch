from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from startorch.sequence import RandInt
from startorch.transformer import Identity, is_transformer_config, setup_transformer

if TYPE_CHECKING:
    import pytest

##################################################
#     Tests for is_transformer_config     #
##################################################


def test_is_transformer_config_true() -> None:
    assert is_transformer_config({OBJECT_TARGET: "startorch.transformer.Identity"})


def test_is_transformer_config_false() -> None:
    assert not is_transformer_config({OBJECT_TARGET: "torch.nn.Identity"})


##############################################
#     Tests for setup_transformer     #
##############################################


def test_setup_transformer_object() -> None:
    transformer = Identity()
    assert setup_transformer(transformer) is transformer


def test_setup_transformer_dict() -> None:
    assert isinstance(
        setup_transformer({OBJECT_TARGET: "startorch.transformer.Identity"}),
        Identity,
    )


def test_setup_transformer_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_transformer({OBJECT_TARGET: "startorch.sequence.RandInt", "low": 0, "high": 10}),
            RandInt,
        )
        assert caplog.messages
