from __future__ import annotations

__all__ = ["RandIntSequenceGenerator", "RandUniformSequenceGenerator"]

from collections.abc import Generator

import torch
from redcat import BatchedTensorSeq

from startorch.random import rand_uniform
from startorch.sequence.base import BaseSequenceGenerator
from startorch.utils.conversion import to_tuple


class RandIntSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequences of uniformly distributed
    integers.

    Args:
    ----
        low (int): Specifies the minimum value (included).
        high (int): Specifies the maximum value (excluded).
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``tuple()``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import RandInt
        >>> generator = RandInt(0, 100)
        >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        low: int,
        high: int,
        feature_size: tuple[int, ...] | list[int] | int = tuple(),
    ) -> None:
        super().__init__()
        if high < low:
            raise ValueError(f"high ({high}) has to be greater or equal to low ({low})")
        self._low = int(low)
        self._high = int(high)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(low={self._low}, "
            f"high={self._high}, feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            torch.randint(
                low=self._low,
                high=self._high,
                size=(batch_size, seq_len) + self._feature_size,
                generator=rng,
            )
        )


class RandUniformSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator to generate sequences by sampling
    values from a uniform distribution.

    Args:
    ----
        low (float, optional): Specifies the minimum value
            (inclusive). Default: ``0.0``
        high (float, optional): Specifies the maximum value
            (exclusive). Default: ``1.0``
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import RandUniform
        >>> generator = RandUniform()
        >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        feature_size: tuple[int, ...] | list[int] | int = 1,
    ) -> None:
        super().__init__()
        self._low = float(low)
        if high < low:
            raise ValueError(f"high ({high}) has to be greater or equal to low ({low})")
        self._high = float(high)
        self._feature_size = to_tuple(feature_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(low={self._low}, "
            f"high={self._high}, feature_size={self._feature_size})"
        )

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            rand_uniform(
                size=(batch_size, seq_len) + self._feature_size,
                low=self._low,
                high=self._high,
                generator=rng,
            )
        )
