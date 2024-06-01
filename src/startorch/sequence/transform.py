r"""Contain the implementation of a sequence generator that generates a
batch of sequences and then transform it."""

from __future__ import annotations

__all__ = ["TransformSequenceGenerator"]

from typing import TYPE_CHECKING

from coola.utils.format import str_indent, str_mapping

from startorch.sequence.wrapper import BaseWrapperSequenceGenerator
from startorch.tensor.transformer.base import (
    BaseTensorTransformer,
    setup_tensor_transformer,
)

if TYPE_CHECKING:
    import torch

    from startorch.sequence.base import BaseSequenceGenerator


class TransformSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator that generates a batch of
    sequences and then transform it.

    Args:
        generator: The sequence generator or its configuration.
        transformer: The tensor/sequence transformer or its configuration.

    Example usage:

    ```pycon

    >>> from startorch.sequence import TransformSequenceGenerator, Full
    >>> from startorch.tensor.transformer import Abs
    >>> generator = TransformSequenceGenerator(
    ...     generator=Full(value=-1, feature_size=()), transformer=Abs()
    ... )
    >>> generator
    TransformSequenceGenerator(
      (generator): FullSequenceGenerator(value=-1, feature_size=())
      (transformer): AbsTensorTransformer()
    )
    >>> generator.generate(seq_len=6, batch_size=2)
    tensor([[1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]])

    ```
    """

    def __init__(
        self,
        generator: BaseSequenceGenerator | dict,
        transformer: BaseTensorTransformer | dict,
    ) -> None:
        super().__init__(generator=generator)
        self._transformer = setup_tensor_transformer(transformer)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping({"generator": self._generator, "transformer": self._transformer})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        sequence = self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
        return self._transformer.transform(tensor=sequence, rng=rng)
