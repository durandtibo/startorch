from __future__ import annotations

from objectory import OBJECT_TARGET

from startorch.sequence import RandUniform, setup_sequence_generator

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
