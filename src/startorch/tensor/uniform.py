r"""Contain the implementation of tensor generators where the values are
sampled from a uniform distribution."""

from __future__ import annotations

__all__ = [
    "AsinhUniformTensorGenerator",
    "LogUniformTensorGenerator",
    "RandAsinhUniformTensorGenerator",
    "RandIntTensorGenerator",
    "RandLogUniformTensorGenerator",
    "RandUniformTensorGenerator",
    "UniformTensorGenerator",
]

import torch
from coola.utils.format import str_indent, str_mapping

from startorch.random import (
    asinh_uniform,
    log_uniform,
    rand_asinh_uniform,
    rand_log_uniform,
    rand_uniform,
    uniform,
)
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator


class AsinhUniformTensorGenerator(BaseTensorGenerator):
    r"""Implement a tensor generator to generate tensors by sampling
    values from an asinh-uniform distribution.

    Args:
        low: A tensor generator (or its configuration) to
            generate the minimum value (inclusive).
        high: A tensor generator (or its configuration) to
            generate the maximum value (exclusive).

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandUniform, AsinhUniform
    >>> generator = AsinhUniform(
    ...     low=RandUniform(low=-1000, high=-100), high=RandUniform(low=100, high=1000)
    ... )
    >>> generator
    AsinhUniformTensorGenerator(
      (low): RandUniformTensorGenerator(low=-1000.0, high=-100.0)
      (high): RandUniformTensorGenerator(low=100.0, high=1000.0)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, low: BaseTensorGenerator | dict, high: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._low = setup_tensor_generator(low)
        self._high = setup_tensor_generator(high)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"low": self._low, "high": self._high}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return asinh_uniform(
            low=self._low.generate(size=size, rng=rng),
            high=self._high.generate(size=size, rng=rng),
            generator=rng,
        )


class LogUniformTensorGenerator(BaseTensorGenerator):
    r"""Implement a tensor generator to generate tensors by sampling
    values from a log-uniform distribution.

    Args:
        low: A tensor generator (or its configuration) to
            generate the minimum value (inclusive).
        high: A tensor generator (or its configuration) to
            generate the maximum value (exclusive).

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandUniform, LogUniform
    >>> generator = LogUniform(
    ...     low=RandUniform(low=0.1, high=1.0), high=RandUniform(low=100, high=1000)
    ... )
    >>> generator
    LogUniformTensorGenerator(
      (low): RandUniformTensorGenerator(low=0.1, high=1.0)
      (high): RandUniformTensorGenerator(low=100.0, high=1000.0)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, low: BaseTensorGenerator | dict, high: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._low = setup_tensor_generator(low)
        self._high = setup_tensor_generator(high)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"low": self._low, "high": self._high}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return log_uniform(
            low=self._low.generate(size=size, rng=rng),
            high=self._high.generate(size=size, rng=rng),
            generator=rng,
        )


class RandAsinhUniformTensorGenerator(BaseTensorGenerator):
    r"""Implement a tensor generator by sampling values from an asinh-
    uniform distribution.

    Args:
        low: The minimum value (inclusive).
        high: The maximum value (exclusive).

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandAsinhUniform
    >>> generator = RandAsinhUniform(low=-1000, high=1000)
    >>> generator
    RandAsinhUniformTensorGenerator(low=-1000.0, high=1000.0)
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self._low = float(low)
        if high < low:
            msg = f"high ({high}) has to be greater or equal to low ({low})"
            raise ValueError(msg)
        self._high = float(high)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(low={self._low}, high={self._high})"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return rand_asinh_uniform(
            size=size,
            low=self._low,
            high=self._high,
            generator=rng,
        )


class RandIntTensorGenerator(BaseTensorGenerator):
    r"""Implement a tensor generator by sampling integer values from a
    uniform distribution.

    Args:
        low: The minimum value (inclusive).
        high: The maximum value (exclusive).

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandInt
    >>> generator = RandInt(low=0, high=10)
    >>> generator
    RandIntTensorGenerator(low=0, high=10)
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, low: int, high: int) -> None:
        super().__init__()
        self._low = int(low)
        if high <= low:
            msg = f"high ({high}) has to be greater than low ({low})"
            raise ValueError(msg)
        self._high = int(high)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(low={self._low}, high={self._high})"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return torch.randint(size=size, low=self._low, high=self._high, generator=rng)


class RandLogUniformTensorGenerator(BaseTensorGenerator):
    r"""Implement a tensor generator to generate tensors by sampling
    values from a log-uniform distribution.

    Args:
        low: The minimum value (inclusive).
        high: The maximum value (exclusive).

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandLogUniform
    >>> generator = RandLogUniform(low=0.1, high=100.0)
    >>> generator
    RandLogUniformTensorGenerator(low=0.1, high=100.0)
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self._low = float(low)
        if high < low:
            msg = f"high ({high}) has to be greater or equal to low ({low})"
            raise ValueError(msg)
        self._high = float(high)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(low={self._low}, high={self._high})"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return rand_log_uniform(
            size=size,
            low=self._low,
            high=self._high,
            generator=rng,
        )


class RandUniformTensorGenerator(BaseTensorGenerator):
    r"""Implement a tensor generator by sampling values from a uniform
    distribution.

    Args:
        low: The minimum value (inclusive).
        high: The maximum value (exclusive).

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandUniform
    >>> generator = RandUniform(low=0, high=10)
    >>> generator
    RandUniformTensorGenerator(low=0.0, high=10.0)
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        super().__init__()
        self._low = float(low)
        if high < low:
            msg = f"high ({high}) has to be greater or equal to low ({low})"
            raise ValueError(msg)
        self._high = float(high)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(low={self._low}, high={self._high})"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return rand_uniform(size=size, low=self._low, high=self._high, generator=rng)


class UniformTensorGenerator(BaseTensorGenerator):
    r"""Implement a tensor generator by sampling values from a uniform
    distribution.

    Args:
        low: A tensor generator (or its configuration) to
            generate the minimum value (inclusive).
        high: A tensor generator (or its configuration) to
            generate the maximum value (exclusive).

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandUniform, Uniform
    >>> generator = UniformTensorGenerator(
    ...     low=RandUniform(low=0, high=2), high=RandUniform(low=8, high=10)
    ... )
    >>> generator
    UniformTensorGenerator(
      (low): RandUniformTensorGenerator(low=0.0, high=2.0)
      (high): RandUniformTensorGenerator(low=8.0, high=10.0)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, low: BaseTensorGenerator | dict, high: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._low = setup_tensor_generator(low)
        self._high = setup_tensor_generator(high)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"low": self._low, "high": self._high}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return uniform(
            low=self._low.generate(size, rng=rng),
            high=self._high.generate(size, rng=rng),
            generator=rng,
        )
