from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from startorch.sequence import RandInt
from startorch.transition import (
    Diagonal,
    is_transition_generator_config,
    setup_transition_generator,
)

if TYPE_CHECKING:
    import pytest

####################################################
#     Tests for is_transition_generator_config     #
####################################################


def test_is_transition_generator_config_true() -> None:
    assert is_transition_generator_config({OBJECT_TARGET: "startorch.transition.Diagonal"})


def test_is_transition_generator_config_false() -> None:
    assert not is_transition_generator_config({OBJECT_TARGET: "torch.nn.Identity"})


################################################
#     Tests for setup_transition_generator     #
################################################


def test_setup_transition_generator_object() -> None:
    generator = Diagonal()
    assert setup_transition_generator(generator) is generator


def test_setup_transition_generator_dict() -> None:
    assert isinstance(
        setup_transition_generator({OBJECT_TARGET: "startorch.transition.Diagonal"}),
        Diagonal,
    )


def test_setup_transition_generator_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_transition_generator(
                {OBJECT_TARGET: "startorch.sequence.RandInt", "low": 0, "high": 10}
            ),
            RandInt,
        )
        assert caplog.messages
