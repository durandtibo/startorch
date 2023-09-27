from __future__ import annotations

import logging

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture

from startorch.example import (
    Hypercube,
    is_example_generator_config,
    setup_example_generator,
)
from startorch.tensor import RandInt

##################################################
#     Tests for is_example_generator_config     #
##################################################


def test_is_example_generator_config_true() -> None:
    assert is_example_generator_config({OBJECT_TARGET: "startorch.example.Hypercube"})


def test_is_example_generator_config_false() -> None:
    assert not is_example_generator_config({OBJECT_TARGET: "torch.nn.Identity"})


##############################################
#     Tests for setup_example_generator     #
##############################################


def test_setup_example_generator_object() -> None:
    generator = Hypercube()
    assert setup_example_generator(generator) is generator


def test_setup_example_generator_dict() -> None:
    assert isinstance(
        setup_example_generator({OBJECT_TARGET: "startorch.example.Hypercube"}),
        Hypercube,
    )


def test_setup_example_generator_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_example_generator(
                {OBJECT_TARGET: "startorch.tensor.RandInt", "low": 0, "high": 10}
            ),
            RandInt,
        )
        assert caplog.messages
