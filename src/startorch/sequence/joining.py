r"""Contain the implementations of sequence generators that generate
sequences by joining generated sequences."""

from __future__ import annotations

__all__ = ["Cat2SequenceGenerator"]

from typing import TYPE_CHECKING

from coola.utils.format import str_indent, str_mapping

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator

if TYPE_CHECKING:
    from redcat import BatchedTensorSeq
    from torch import Generator


class Cat2SequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator that concatenate two sequences
    along the sequence dimension.

    ``ouput = [sequence1, sequence2]``

    Args:
        generator1: Specifies the first sequence generator or its
            configuration.
        generator2: Specifies the second sequence generator or its
            configuration.
        changepoint: Specifies the change point generator or its
            configuration.

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.sequence import Cat2, RandUniform, RandNormal
    >>> from startorch.tensor import RandInt
    >>> generator = Cat2(
    ...     generator1=RandUniform(), generator2=RandNormal(), changepoint=RandInt(0, 12)
    ... )
    >>> generator
    Cat2SequenceGenerator(
      (generator1): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
      (generator2): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
      (changepoint): RandIntTensorGenerator(low=0, high=12)
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        generator1: BaseSequenceGenerator | dict,
        generator2: BaseSequenceGenerator | dict,
        changepoint: BaseTensorGenerator | dict,
    ) -> None:
        super().__init__()
        self._generator1 = setup_sequence_generator(generator1)
        self._generator2 = setup_sequence_generator(generator2)
        self._changepoint = setup_tensor_generator(changepoint)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "generator1": self._generator1,
                    "generator2": self._generator2,
                    "changepoint": self._changepoint,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        changepoint = max(min(int(self._changepoint.generate((1,), rng=rng).item()), seq_len), 0)
        return self._generator1.generate(
            seq_len=changepoint, batch_size=batch_size, rng=rng
        ).cat_along_seq(
            self._generator2.generate(seq_len=seq_len - changepoint, batch_size=batch_size, rng=rng)
        )
