from __future__ import annotations

import logging

import torch
from redcat import BatchedTensorSeq
from torch import Tensor

from startorch.random import rand_uniform
from startorch.sequence import RandUniform as SRandUniform
from startorch.tensor import RandUniform as TRandUniform

logger = logging.getLogger(__name__)


def check_random() -> None:
    logger.info("Checking startorch.random package...")
    assert isinstance(rand_uniform(low=0.0, high=10.0, size=(4, 12)), torch.Tensor)


def check_sequence() -> None:
    logger.info("Checking startorch.sequence package...")
    generator = SRandUniform()
    seq = generator.generate(seq_len=12, batch_size=4)
    assert isinstance(seq, BatchedTensorSeq)
    assert seq.batch_size == 4
    assert seq.seq_len == 12


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
