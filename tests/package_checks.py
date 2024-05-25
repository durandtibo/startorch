from __future__ import annotations

import logging

import torch
from torch import Tensor

from startorch import constants as ct
from startorch.example import CirclesClassification
from startorch.random import rand_uniform
from startorch.sequence import RandUniform as SRandUniform
from startorch.tensor import RandUniform as TRandUniform
from startorch.timeseries import SequenceTimeSeries

logger = logging.getLogger(__name__)


def check_example() -> None:
    logger.info("Checking startorch.example package...")
    data = CirclesClassification().generate(batch_size=12)
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data[ct.TARGET], torch.Tensor)
    assert data[ct.TARGET].shape == (12,)
    assert data[ct.TARGET].dtype == torch.long
    assert isinstance(data[ct.FEATURE], torch.Tensor)
    assert data[ct.FEATURE].shape == (12, 2)
    assert data[ct.FEATURE].dtype == torch.float


def check_random() -> None:
    logger.info("Checking startorch.random package...")
    assert isinstance(rand_uniform(low=0.0, high=10.0, size=(4, 12)), torch.Tensor)


def check_sequence() -> None:
    logger.info("Checking startorch.sequence package...")
    seq = SRandUniform().generate(seq_len=12, batch_size=4)
    assert isinstance(seq, torch.Tensor)
    assert seq.shape == (4, 12, 1)


def check_tensor() -> None:
    logger.info("Checking startorch.tensor package...")
    tensor = TRandUniform().generate(size=(4, 12))
    assert isinstance(tensor, Tensor)
    assert tensor.shape == (4, 12)


def check_timeseries() -> None:
    logger.info("Checking startorch.timeseries package...")
    data = SequenceTimeSeries({"value": SRandUniform(), "time": SRandUniform()}).generate(
        seq_len=12, batch_size=4
    )
    assert isinstance(data, dict)
    assert len(data) == 2
    assert isinstance(data["value"], torch.Tensor)
    assert data["value"].shape == (4, 12, 1)
    assert data["value"].dtype == torch.float
    assert isinstance(data["time"], torch.Tensor)
    assert data["time"].shape == (4, 12, 1)
    assert data["time"].dtype == torch.float


def main() -> None:
    check_random()
    check_sequence()
    check_tensor()
    check_timeseries()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
