from __future__ import annotations

__all__ = [
    "AsinhUniformSequenceGenerator",
    "LogUniformSequenceGenerator",
    "RandAsinhUniformSequenceGenerator",
    "RandIntSequenceGenerator",
    "RandLogUniformSequenceGenerator",
    "RandUniformSequenceGenerator",
    "UniformSequenceGenerator",
]

from collections.abc import Generator

import torch
from coola.utils import str_indent, str_mapping
from redcat import BatchedTensorSeq

from startorch.random import (
    asinh_uniform,
    log_uniform,
    rand_asinh_uniform,
    rand_log_uniform,
    rand_uniform,
    uniform,
)
from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.utils.conversion import to_tuple


class AsinhUniformSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator to generate sequences by sampling
    values from an asinh-uniform distribution.

    Args:
    ----
        low (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the minimum
            value (inclusive).
        high (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the maximum
            value (exclusive).

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import AsinhUniform, RandUniform
        >>> generator = AsinhUniform(low=RandUniform(-1.0, 0.0), high=RandUniform(0.0, 1.0))
        >>> generator
        AsinhUniformSequenceGenerator(
          (low): RandUniformSequenceGenerator(low=-1.0, high=0.0, feature_size=(1,))
          (high): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        low: BaseSequenceGenerator | dict,
        high: BaseSequenceGenerator | dict,
    ) -> None:
        super().__init__()
        self._low = setup_sequence_generator(low)
        self._high = setup_sequence_generator(high)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"low": self._low, "high": self._high}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            asinh_uniform(
                low=self._low.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                high=self._high.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                generator=rng,
            )
        )


class LogUniformSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator to generate sequences by sampling
    values from a log-uniform distribution.

    Args:
    ----
        low (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the minimum
            value (inclusive).
        high (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the maximum
            value (exclusive).

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import LogUniform, RandUniform
        >>> generator = LogUniform(low=RandUniform(0.0, 1.0), high=RandUniform(5.0, 10.0))
        >>> generator
        LogUniformSequenceGenerator(
          (low): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
          (high): RandUniformSequenceGenerator(low=5.0, high=10.0, feature_size=(1,))
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        low: BaseSequenceGenerator | dict,
        high: BaseSequenceGenerator | dict,
    ) -> None:
        super().__init__()
        self._low = setup_sequence_generator(low)
        self._high = setup_sequence_generator(high)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"low": self._low, "high": self._high}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            log_uniform(
                low=self._low.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                high=self._high.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                generator=rng,
            )
        )


class RandAsinhUniformSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator to generate sequences by sampling
    values from an asinh-uniform distribution.

    Args:
    ----
        low (float): Specifies the minimum value (inclusive).
        high (float): Specifies the maximum value (exclusive).
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Raises:
    ------
        ValueError if ``high`` is lower than ``low``.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import RandAsinhUniform
        >>> generator = RandAsinhUniform(low=1.0, high=10.0)
        >>> generator
        RandAsinhUniformSequenceGenerator(low=1.0, high=10.0, feature_size=(1,))
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        low: float,
        high: float,
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
            rand_asinh_uniform(
                size=(batch_size, seq_len) + self._feature_size,
                low=self._low,
                high=self._high,
                generator=rng,
            )
        )


class RandIntSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequences of uniformly distributed
    integers.

    Args:
    ----
        low (int): Specifies the minimum value (included).
        high (int): Specifies the maximum value (excluded).
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``tuple()``

    Raises:
    ------
        ValueError if ``high`` is lower than ``low``.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import RandInt
        >>> generator = RandInt(0, 100)
        >>> generator
        RandIntSequenceGenerator(low=0, high=100, feature_size=())
        >>> generator.generate(seq_len=12, batch_size=4)
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


class RandLogUniformSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator to generate sequences by sampling
    values from a log-uniform distribution.

    Args:
    ----
        low (float): Specifies the minimum value (inclusive).
        high (float): Specifies the maximum value (exclusive).
        feature_size (tuple or list or int, optional): Specifies the
            feature size. Default: ``1``

    Raises:
    ------
        ValueError if ``high`` is lower than ``low``.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import RandLogUniform
        >>> generator = RandLogUniform(low=1.0, high=10.0)
        >>> generator
        RandLogUniformSequenceGenerator(low=1.0, high=10.0, feature_size=(1,))
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        low: float,
        high: float,
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
            rand_log_uniform(
                size=(batch_size, seq_len) + self._feature_size,
                low=self._low,
                high=self._high,
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

    Raises:
    ------
        ValueError if ``high`` is lower than ``low``.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import RandUniform
        >>> generator = RandUniform()
        >>> generator
        RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        >>> generator.generate(seq_len=12, batch_size=4)
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


class UniformSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator to generate sequences by sampling
    values from a uniform distribution.

    Args:
        low (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the minimum
            value (inclusive).
        high (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) to generate the maximum
            value (exclusive).

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import Uniform, RandUniform
        >>> generator = Uniform(low=RandUniform(-1.0, 0.0), high=RandUniform(0.0, 1.0))
        >>> generator
        UniformSequenceGenerator(
          (low): RandUniformSequenceGenerator(low=-1.0, high=0.0, feature_size=(1,))
          (high): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self, low: BaseSequenceGenerator | dict, high: BaseSequenceGenerator | dict
    ) -> None:
        super().__init__()
        self._low = setup_sequence_generator(low)
        self._high = setup_sequence_generator(high)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"low": self._low, "high": self._high}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return BatchedTensorSeq(
            uniform(
                low=self._low.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                high=self._high.generate(seq_len=seq_len, batch_size=batch_size, rng=rng).data,
                generator=rng,
            )
        )
