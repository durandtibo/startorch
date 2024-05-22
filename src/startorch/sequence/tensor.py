r"""Contain the implementation of sequence generator that generates
sequences from a tensor generator."""

from __future__ import annotations

__all__ = ["TensorSequenceGenerator"]

from typing import TYPE_CHECKING

from coola.utils.format import str_indent, str_mapping

from startorch.sequence.base import BaseSequenceGenerator
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator
from startorch.utils.conversion import to_tuple

if TYPE_CHECKING:
    import torch


class TensorSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a sequence generator to generate sequences from a
    tensor generator.

    Args:
        tensor: A tensor generator (or its configuration).
        feature_size: The feature size.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.sequence import TensorSequence
    >>> from startorch.tensor import RandUniform
    >>> generator = TensorSequence(RandUniform())
    >>> generator
    TensorSequenceGenerator(
      (tensor): RandUniformTensorGenerator(low=0.0, high=1.0)
      (feature_size): ()
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        tensor: BaseTensorGenerator | dict,
        feature_size: tuple[int, ...] | list[int] | int = (),
    ) -> None:
        super().__init__()
        self._tensor = setup_tensor_generator(tensor)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"tensor": self._tensor, "feature_size": self._feature_size}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._tensor.generate(size=(batch_size, seq_len, *self._feature_size), rng=rng)
