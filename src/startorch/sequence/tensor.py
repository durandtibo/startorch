from __future__ import annotations

__all__ = ["TensorSequenceGenerator"]


from coola.utils.format import str_indent, str_mapping
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator
from startorch.utils.conversion import to_tuple


class TensorSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator to generate sequences from a
    tensor generator.

    Args:
        tensor (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration).
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import TensorSequence
        >>> from startorch.tensor import RandUniform
        >>> generator = TensorSequence(RandUniform())
        >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        tensor: BaseTensorGenerator | dict,
        feature_size: tuple[int, ...] | list[int] | int = tuple(),
    ) -> None:
        super().__init__()
        self._tensor = setup_tensor_generator(tensor)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"tensor": self._tensor, "feature_size": self._feature_size}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            self._tensor.generate(size=(batch_size, seq_len) + self._feature_size, rng=rng)
        )
