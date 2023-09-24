from __future__ import annotations

from torch import Generator, Tensor

from startorch.tensor import BaseWrapperTensorGenerator, RandUniform
from startorch.utils.seed import get_torch_generator

################################################
#     Tests for BaseWrapperTensorGenerator     #
################################################


class FakeWrapper(BaseWrapperTensorGenerator):
    r"""Generates a fake class to test
    ``BaseWrapperTensorGenerator``."""

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._generator.generate(size, rng=rng)


def test_base_wrapper_str() -> None:
    assert str(FakeWrapper(RandUniform())).startswith("FakeWrapper(")


def test_base_wrapper_generate_same_random_seed() -> None:
    generator = FakeWrapper(RandUniform())
    assert generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(1))
    )


def test_base_wrapper_generate_different_random_seeds() -> None:
    generator = FakeWrapper(RandUniform())
    assert not generator.generate(size=(4, 12), rng=get_torch_generator(1)).equal(
        generator.generate(size=(4, 12), rng=get_torch_generator(2))
    )
