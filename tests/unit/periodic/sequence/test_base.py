from objectory import OBJECT_TARGET

from startorch.periodic.sequence import Repeat, setup_periodic_sequence_generator
from startorch.sequence import RandUniform

#######################################################
#     Tests for setup_periodic_sequence_generator     #
#######################################################


def test_setup_periodic_sequence_generator_object() -> None:
    generator = Repeat(RandUniform())
    assert setup_periodic_sequence_generator(generator) is generator


def test_setup_periodic_sequence_generator_dict() -> None:
    assert isinstance(
        setup_periodic_sequence_generator(
            {
                OBJECT_TARGET: "startorch.periodic.sequence.Repeat",
                "sequence": {OBJECT_TARGET: "startorch.sequence.RandUniform"},
            }
        ),
        Repeat,
    )
