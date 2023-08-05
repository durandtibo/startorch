from objectory import OBJECT_TARGET

from startorch.sequence import RandUniform, UniformCategorical
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
