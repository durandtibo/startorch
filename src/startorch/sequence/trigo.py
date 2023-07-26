from __future__ import annotations

__all__ = [
    "AcoshSequenceGenerator",
    "AsinhSequenceGenerator",
    "CoshSequenceGenerator",
    "SinhSequenceGenerator",
]

from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.wrapper import BaseWrapperSequenceGenerator


class AcoshSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that computes the inverse
    hyperbolic cosine (arccosh) of each value.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import Acosh, RandUniform
        >>> generator = Acosh(RandUniform())
        >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).acosh()


class AsinhSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that computes the inverse
    hyperbolic sine (arcsinh) of each value.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import Asinh, RandUniform
        >>> generator = Asinh(RandUniform())
        >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).asinh()


class CoshSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that computes the hyperbolic
    cosine (cosh) of each value.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import Cosh, RandUniform
        >>> generator = Cosh(RandUniform())
        >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).cosh()


class SinhSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator that computes the hyperbolic sine
    (sinh) of each value.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import Sinh, RandUniform
        >>> generator = Sinh(RandUniform())
        >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).sinh()
