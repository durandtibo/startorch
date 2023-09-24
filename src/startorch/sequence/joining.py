from __future__ import annotations

__all__ = ["Cat2SequenceGenerator"]


from coola.utils.format import str_indent, str_mapping
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator


class Cat2SequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator that concatenate two sequences.

    ``sequence = [sequence_1, sequence_2]``

    Args:
    ----
        generator1 (``BaseSequenceGenerator`` or dict): Specifies the
            first sequence generator or its configuration.
        generator2 (``BaseSequenceGenerator`` or dict): Specifies the
            second sequence generator or its configuration.
        changepoint (``BaseTensorGenerator`` or dict): Specifies the
            change point generator or its configuration.

    Example usage:

    .. code-block:: pycon

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
        tensor([[...]], batch_dim=0, seq_dim=1)
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
