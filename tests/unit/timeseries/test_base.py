import logging

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture

from startorch.sequence import RandInt, RandUniform, UniformCategorical
from startorch.timeseries import TimeSeries, setup_timeseries_generator

################################################
#     Tests for setup_timeseries_generator     #
################################################


def test_setup_timeseries_generator_object() -> None:
    generator = TimeSeries({"value": UniformCategorical(num_categories=10), "time": RandUniform()})
    assert setup_timeseries_generator(generator) is generator


def test_setup_timeseries_generator_dict() -> None:
    assert isinstance(
        setup_timeseries_generator(
            {
                OBJECT_TARGET: "startorch.timeseries.TimeSeries",
                "sequences": {
                    "value": {
                        OBJECT_TARGET: "startorch.sequence.UniformCategorical",
                        "num_categories": 10,
                    },
                    "time": {OBJECT_TARGET: "startorch.sequence.RandUniform"},
                },
            }
        ),
        TimeSeries,
    )


def test_setup_timeseries_generator_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_timeseries_generator(
                {OBJECT_TARGET: "startorch.sequence.RandInt", "low": 0, "high": 10}
            ),
            RandInt,
        )
        assert caplog.messages
