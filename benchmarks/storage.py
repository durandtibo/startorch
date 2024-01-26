from __future__ import annotations

import logging
from pathlib import Path

import torch
from gravitorch.utils.io import load_pickle, save_pickle
from gravitorch.utils.path import get_human_readable_file_size
from gravitorch.utils.timing import timeblock
from safetensors.torch import load_file, save_file
from torch import Tensor

logger = logging.getLogger(__name__)


def generate_tensors(
    num_examples: int = 1000, feature_size: int = 128, num_keys: int = 1
) -> dict[str, Tensor]:
    logger.info("Generating tensors...")
    with timeblock("generate tensors | time: {time}"):
        tensors = {}
        for i in range(num_keys):
            tensors[str(i)] = torch.randn(num_examples, feature_size)
    return tensors


def benchmark_save_safetensors(tensors: dict[str, Tensor], path: Path) -> None:
    logger.info("Saving tensors with safetensors...")
    with timeblock("save safetensors | time: {time}"):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, path)


def benchmark_load_safetensors(path: Path) -> dict[str, Tensor]:
    logger.info(f"File size: {get_human_readable_file_size(path)}")
    logger.info(f"Loading tensors with safetensors from {path}...")
    tensors = {}
    with timeblock("load safetensors | time: {time}"):
        tensors = load_file(path)
        # with safe_open(path, framework="pt", device="cpu") as f:
        #     for key in f.keys():
        #         tensors[key] = f.get_tensor(key)
        print(type(tensors))
        print([type(t) for t in tensors.values()])
        print([t.shape for t in tensors.values()])
    return tensors


def benchmark_save_pickle(tensors: dict[str, Tensor], path: Path) -> None:
    logger.info("Saving tensors with pickle...")
    with timeblock("save pickle | time: {time}"):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(tensors, path)


def benchmark_load_pickle(path: Path) -> dict[str, Tensor]:
    logger.info(f"File size: {get_human_readable_file_size(path)}")
    logger.info(f"Loading tensors with pickle from {path}...")
    with timeblock("load pickle | time: {time}"):
        tensors = load_pickle(path)
    return tensors


def main() -> None:
    path = Path("tmp/torch/")
    n = 5
    # for i in range(n):
    #     tensors = generate_tensors(num_examples=int(1e6), feature_size=256, num_keys=3)
    #     benchmark_save_safetensors(tensors, path.joinpath(f"safetensors/tensors_{i}.safetensors"))
    #     benchmark_save_pickle(tensors, path.joinpath(f"pickle/tensors_{i}.pickle"))

    with timeblock("all load safetensors | time: {time}"):
        for i in range(n):
            benchmark_load_safetensors(path.joinpath(f"safetensors/tensors_{i}.safetensors"))

    with timeblock("all load pickle | time: {time}"):
        for i in range(n):
            benchmark_load_pickle(path.joinpath(f"pickle/tensors_{i}.pickle"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
