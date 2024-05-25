r"""Contain functions to benchmark data loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from batchtensor.nested import split_along_batch
from coola.utils import str_mapping
from iden.dataset import BaseDataset, load_from_uri
from iden.dataset.generator import VanillaDatasetGenerator
from iden.shard.generator import (
    BaseShardGenerator,
    PickleShardGenerator,
    ShardDictGenerator,
    ShardTupleGenerator,
    TorchSafetensorsShardGenerator,
    TorchShardGenerator,
)
from iden.shard.utils import ShardIterable
from iden.utils.format import human_time
from iden.utils.time import sync_perf_counter

from startorch import constants as ct
from startorch.example import BaseExampleGenerator, TimeSeriesExampleGenerator
from startorch.integration.iden.data.generator import ExampleDataGenerator
from startorch.sequence import RandUniformSequenceGenerator
from startorch.tensor import FullTensorGenerator
from startorch.timeseries import SequenceTimeSeriesGenerator

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    r"""Define the dataset configuration.

    Args:
        name: The dataset name.
        path: The dataset path.
        batch_size: The batch size of the data in a shard.
        num_shards: The number of shards per dataset split.
        shard_generator_type: The class to use to generate a shard.
        splits: The dataset splits.
    """

    name: str
    path: Path
    batch_size: int
    num_shards: int
    shard_generator_type: type[BaseShardGenerator]
    splits: tuple[str, ...] = ("train", "val", "test")


def benchmark_data_loading(dataset: BaseDataset) -> float:
    r"""Benchmark the time to load the data from the dataset.

    Args:
        dataset: The dataset to benchmark.

    Returns:
        The data loading time in second.
    """
    start_time = sync_perf_counter()
    total = 0
    for split in dataset.get_splits():
        shards = dataset.get_shards(split)
        for data in ShardIterable(shards):
            for batch in split_along_batch(data, split_size_or_sections=128):
                total += batch[ct.TARGET].shape[0]
    end_time = sync_perf_counter()
    logger.info(f"total: {total:,}")
    logger.info(f"total time: {human_time(sync_perf_counter() - start_time)}")
    return end_time - start_time


def get_example_generator() -> BaseExampleGenerator:
    r"""Get an example generator.

    Returns:
        An example generator.
    """
    return TimeSeriesExampleGenerator(
        timeseries=SequenceTimeSeriesGenerator(
            generators={
                ct.TARGET: RandUniformSequenceGenerator(),
                ct.VALUE: RandUniformSequenceGenerator(),
                ct.TIME: RandUniformSequenceGenerator(),
            }
        ),
        seq_len=FullTensorGenerator(value=256),
    )


def generate_dataset(config: DatasetConfig) -> None:
    r"""Generate the dataset from its configuration.

    Args:
        config: The dataset configuration.
    """
    path_uri = config.path.joinpath("uri")
    path_data = config.path.joinpath("data")

    generator = VanillaDatasetGenerator(
        path_uri=path_uri,
        shards=ShardDictGenerator(
            path_uri=path_uri,
            shards={
                split: ShardTupleGenerator(
                    shard=config.shard_generator_type(
                        data=ExampleDataGenerator(
                            example=get_example_generator(),
                            batch_size=config.batch_size,
                            random_seed=13860096089045506684,
                        ),
                        path_uri=path_uri.joinpath(split).joinpath("shards"),
                        path_shard=path_data.joinpath(split).joinpath("shards"),
                    ),
                    num_shards=config.num_shards,
                    path_uri=path_uri.joinpath(split),
                )
                for split in config.splits
            },
        ),
        assets=ShardDictGenerator(path_uri=path_uri.joinpath("assets"), shards={}),
    )
    logger.info(f"dataset generator:\n{generator}")
    generator.generate("dataset")


def get_dataset(config: DatasetConfig) -> BaseDataset:
    r"""Return a dataset based on a path.

    If the dataset does not exist, it is automatically generated.

    Args:
        config: The dataset configuration.
    """
    uri_file = config.path.joinpath("uri/dataset")
    if not uri_file.is_file():
        generate_dataset(config=config)

    uri = uri_file.as_uri()
    logger.info(f"loading dataset from {uri}")
    return load_from_uri(uri)


def main() -> None:
    r"""Implement the main function."""
    path = Path.cwd().joinpath("tmp/dataset")

    num_shards = 5
    batch_size = 1000000
    splits = ("train",)
    configs = [
        DatasetConfig(
            name="safetensors1",
            batch_size=batch_size,
            num_shards=num_shards,
            path=path.joinpath("safetensors1"),
            shard_generator_type=TorchSafetensorsShardGenerator,
            splits=splits,
        ),
        DatasetConfig(
            name="pickle1",
            batch_size=batch_size,
            num_shards=num_shards,
            path=path.joinpath("pickle1"),
            shard_generator_type=PickleShardGenerator,
            splits=splits,
        ),
        DatasetConfig(
            name="torch1",
            batch_size=batch_size,
            num_shards=num_shards,
            path=path.joinpath("torch1"),
            shard_generator_type=TorchShardGenerator,
            splits=splits,
        ),
    ]
    data_loading_times = {}
    for config in configs:
        logger.info(f"config: {config}")

        dataset = get_dataset(config=config)
        logger.info(f"dataset:\n{dataset}")

        data_loading_times[config.name] = benchmark_data_loading(dataset)

    min_value = min(data_loading_times.values())
    logger.info(
        "\n"
        + str_mapping(
            {
                name: human_time(t) + f"\tx{t/min_value:.2f}"
                for name, t in data_loading_times.items()
            }
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
