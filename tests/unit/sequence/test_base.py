from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from startorch.sequence import (
    RandUniform,
    is_sequence_generator_config,
    setup_sequence_generator,
)
from startorch.tensor import RandInt

if TYPE_CHECKING:
    import pytest

##################################################
#     Tests for is_sequence_generator_config     #
##################################################


def test_is_sequence_generator_config_true() -> None:
    assert is_sequence_generator_config({OBJECT_TARGET: "startorch.sequence.RandUniform"})


def test_is_sequence_generator_config_false() -> None:
    assert not is_sequence_generator_config({OBJECT_TARGET: "torch.nn.Identity"})


##############################################
#     Tests for setup_sequence_generator     #
##############################################


def test_setup_sequence_generator_object() -> None:
    generator = RandUniform()
    assert setup_sequence_generator(generator) is generator


def test_setup_sequence_generator_dict() -> None:
    assert isinstance(
        setup_sequence_generator({OBJECT_TARGET: "startorch.sequence.RandUniform"}),
        RandUniform,
    )


def test_setup_sequence_generator_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_sequence_generator(
                {OBJECT_TARGET: "startorch.tensor.RandInt", "low": 0, "high": 10}
            ),
            RandInt,
        )
        assert caplog.messages
