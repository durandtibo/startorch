r"""Contain the implementation of example generator that concatenates
the outputs of multiple example generators."""

from __future__ import annotations

__all__ = ["ConcatenateExampleGenerator"]

from typing import TYPE_CHECKING

from coola.utils import str_indent, str_sequence

from startorch.example.base import BaseExampleGenerator, setup_example_generator

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import torch


class ConcatenateExampleGenerator(BaseExampleGenerator):
    r"""Implement an example generator that concatenates the outputs of
    multiple example generators.

    Note that the last value is used if there are duplicated keys.

    Args:
        generators: The example generators or their configurations.

    Example usage:

    ```pycon

    >>> from startorch.example import TensorExampleGenerator, Concatenate
    >>> from startorch.tensor import RandInt, RandUniform
    >>> generator = Concatenate(
    ...     [
    ...         TensorExampleGenerator(
    ...             generators={"value": RandUniform(), "time": RandUniform()},
    ...             size=(6,),
    ...         ),
    ...         TensorExampleGenerator(
    ...             generators={"label": RandInt(0, 10)},
    ...             size=(),
    ...         ),
    ...     ]
    ... )
    >>> generator
    ConcatenateExampleGenerator(
      (0): TensorExampleGenerator(
          (value): RandUniformTensorGenerator(low=0.0, high=1.0)
          (time): RandUniformTensorGenerator(low=0.0, high=1.0)
          (size): (6,)
        )
      (1): TensorExampleGenerator(
          (label): RandIntTensorGenerator(low=0, high=10)
          (size): ()
        )
    )
    >>> generator.generate(batch_size=10)
    {'value': tensor([[...]]), 'time': tensor([[...]]), 'label': tensor([...])}

    ```
    """

    def __init__(
        self,
        generators: Sequence[BaseExampleGenerator | dict],
    ) -> None:
        super().__init__()
        self._generators = [setup_example_generator(generator) for generator in generators]

    def __repr__(self) -> str:
        args = str_indent(str_sequence(self._generators))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        out = {}
        for generator in self._generators:
            out |= generator.generate(batch_size, rng)
        return out
