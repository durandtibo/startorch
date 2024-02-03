from __future__ import annotations

import logging

import torch
from torch import Tensor

from startorch import constants as ct
from startorch.example import CirclesClassification
from startorch.random import rand_uniform
from startorch.sequence import RandUniform as SRandUniform
from startorch.tensor import RandUniform as TRandUniform

logger = logging.getLogger(__name__)


def check_example() -> None:
    logger.info("Checking startorch.example package...")
    generator = CirclesClassification()
    data = generator.generate(batch_size=12)
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
    generator = SRandUniform()
    seq = generator.generate(seq_len=12, batch_size=4)
    assert isinstance(seq, torch.Tensor)
    assert seq.shape == (4, 12, 1)


def check_tensor() -> None:
    logger.info("Checking startorch.tensor package...")
    generator = TRandUniform()
    seq = generator.generate(size=(4, 12))
    assert isinstance(seq, Tensor)
    assert seq.shape == (4, 12)


def main() -> None:
    check_random()
    check_sequence()
    check_tensor()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
