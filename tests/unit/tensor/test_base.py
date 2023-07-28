from __future__ import annotations

import logging

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture

from startorch.sequence import RandInt
from startorch.tensor import RandUniform, setup_tensor_generator

############################################
#     Tests for setup_tensor_generator     #
############################################


def test_setup_tensor_generator_object() -> None:
    generator = RandUniform()
    assert setup_tensor_generator(generator) is generator


def test_setup_tensor_generator_dict() -> None:
    assert isinstance(
        setup_tensor_generator({OBJECT_TARGET: "startorch.tensor.RandUniform"}),
        RandUniform,
    )


def test_setup_tensor_generator_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_tensor_generator(
                {OBJECT_TARGET: "startorch.sequence.RandInt", "low": 0, "high": 10}
            ),
            RandInt,
        )
        assert caplog.messages
