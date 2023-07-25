from __future__ import annotations

from objectory import OBJECT_TARGET

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
