from __future__ import annotations

import logging

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture

from startorch.periodic.timeseries import Repeat, setup_periodic_timeseries_generator
from startorch.sequence import RandUniform
from startorch.timeseries import TimeSeries

#########################################################
#     Tests for setup_periodic_timeseries_generator     #
#########################################################


def test_setup_periodic_timeseries_generator_object() -> None:
    generator = Repeat(TimeSeries({"value": RandUniform(), "time": RandUniform()}))
    assert setup_periodic_timeseries_generator(generator) is generator


def test_setup_periodic_timeseries_generator_dict() -> None:
    assert isinstance(
        setup_periodic_timeseries_generator(
            {
                OBJECT_TARGET: "startorch.periodic.timeseries.Repeat",
                "generator": {
                    OBJECT_TARGET: "startorch.timeseries.TimeSeries",
                    "sequences": {
                        "value": {OBJECT_TARGET: "startorch.sequence.RandUniform"},
                        "time": {OBJECT_TARGET: "startorch.sequence.RandUniform"},
                    },
                },
            }
        ),
        Repeat,
    )


def test_setup_periodic_timeseries_generator_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_periodic_timeseries_generator(
                {OBJECT_TARGET: "startorch.sequence.RandUniform", "low": 0, "high": 10}
            ),
            RandUniform,
        )
        assert caplog.messages
