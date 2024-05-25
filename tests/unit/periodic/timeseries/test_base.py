from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from startorch.periodic.timeseries import (
    Repeat,
    is_periodic_timeseries_generator_config,
    setup_periodic_timeseries_generator,
)
from startorch.sequence import RandUniform
from startorch.timeseries import SequenceTimeSeries

if TYPE_CHECKING:
    import pytest

#############################################################
#     Tests for is_periodic_timeseries_generator_config     #
#############################################################


def test_is_periodic_timeseries_generator_config_true() -> None:
    assert is_periodic_timeseries_generator_config(
        {
            OBJECT_TARGET: "startorch.periodic.timeseries.Repeat",
            "generator": {
                OBJECT_TARGET: "startorch.timeseries.SequenceTimeSeries",
                "generators": {
                    "value": {OBJECT_TARGET: "startorch.sequence.RandUniform"},
                    "time": {OBJECT_TARGET: "startorch.sequence.RandUniform"},
                },
            },
        }
    )


def test_is_periodic_timeseries_generator_config_false() -> None:
    assert not is_periodic_timeseries_generator_config({OBJECT_TARGET: "torch.nn.Identity"})


#########################################################
#     Tests for setup_periodic_timeseries_generator     #
#########################################################


def test_setup_periodic_timeseries_generator_object() -> None:
    generator = Repeat(SequenceTimeSeries({"value": RandUniform(), "time": RandUniform()}))
    assert setup_periodic_timeseries_generator(generator) is generator


def test_setup_periodic_timeseries_generator_dict() -> None:
    assert isinstance(
        setup_periodic_timeseries_generator(
            {
                OBJECT_TARGET: "startorch.periodic.timeseries.Repeat",
                "generator": {
                    OBJECT_TARGET: "startorch.timeseries.SequenceTimeSeries",
                    "generators": {
                        "value": {OBJECT_TARGET: "startorch.sequence.RandUniform"},
                        "time": {OBJECT_TARGET: "startorch.sequence.RandUniform"},
                    },
                },
            }
        ),
        Repeat,
    )


def test_setup_periodic_timeseries_generator_incorrect_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_periodic_timeseries_generator(
                {OBJECT_TARGET: "startorch.sequence.RandUniform", "low": 0, "high": 10}
            ),
            RandUniform,
        )
        assert caplog.messages
