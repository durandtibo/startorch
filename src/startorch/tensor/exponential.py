r"""Contain the implementation of tensor generators where the values are
sampled from an Exponential distribution."""

from __future__ import annotations

__all__ = [
    "ExponentialTensorGenerator",
    "RandExponentialTensorGenerator",
    "RandTruncExponentialTensorGenerator",
    "TruncExponentialTensorGenerator",
]

from typing import TYPE_CHECKING

from coola.utils.format import str_indent, str_mapping

from startorch.random import (
    exponential,
    rand_exponential,
    rand_trunc_exponential,
    trunc_exponential,
)
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator

if TYPE_CHECKING:
    import torch


class ExponentialTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate tensors by sampling values from an
    Exponential distribution.

    The rates of the Exponential distribution are generated by the
    rate generator. The rate generator should return the rate for each
    value in the sequence.

    Args:
        rate: The rate generator or its configuration.
            The rate generator should return valid rate values.

    Example usage:

    ```pycon

    >>> from startorch.tensor import Exponential, RandUniform
    >>> generator = Exponential(rate=RandUniform(low=1.0, high=100.0))
    >>> generator
    ExponentialTensorGenerator(
      (rate): RandUniformTensorGenerator(low=1.0, high=100.0)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, rate: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._rate = setup_tensor_generator(rate)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"rate": self._rate}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return exponential(self._rate.generate(size=size, rng=rng), generator=rng)


class RandExponentialTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate sequences by sampling values from
    an Exponential distribution.

    Args:
        rate: The rate of the Exponential distribution.

    Raises:
        ValueError: if ``rate`` is not a positive number.

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandExponential
    >>> generator = RandExponential(rate=1.0)
    >>> generator
    RandExponentialTensorGenerator(rate=1.0)
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, rate: float = 1.0) -> None:
        super().__init__()
        if rate <= 0:
            msg = f"rate has to be greater than 0 (received: {rate})"
            raise ValueError(msg)
        self._rate = float(rate)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(rate={self._rate})"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return rand_exponential(size=size, rate=self._rate, generator=rng)


class RandTruncExponentialTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate sequences by sampling values from a
    truncated Exponential distribution.

    Args:
        rate: The rate of the Exponential distribution.
        max_value: The maximum value.

    Raises:
        ValueError: if ``rate`` is not a positive number.
        ValueError: if ``max_value`` is not a positive number.

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandTruncExponential
    >>> generator = RandTruncExponential(rate=1.0, max_value=3.0)
    >>> generator
    RandTruncExponentialTensorGenerator(rate=1.0, max_value=3.0)
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, rate: float = 1.0, max_value: float = 5.0) -> None:
        super().__init__()
        if rate <= 0:
            msg = f"rate has to be greater than 0 (received: {rate})"
            raise ValueError(msg)
        self._rate = float(rate)
        if max_value <= 0:
            msg = f"max_value has to be greater than 0 (received: {max_value})"
            raise ValueError(msg)
        self._max_value = float(max_value)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(rate={self._rate}, max_value={self._max_value})"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return rand_trunc_exponential(
            size=size,
            rate=self._rate,
            max_value=self._max_value,
            generator=rng,
        )


class TruncExponentialTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate sequence by sampling values from an
    Exponential distribution.

    Args:
        rate: A sequence generator (or its configuration) to
            generate the rate.
        max_value: A sequence generator (or its
            configuration) to generate the maximum value (excluded).

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandUniform, TruncExponential
    >>> generator = TruncExponential(
    ...     rate=RandUniform(low=1.0, high=10.0),
    ...     max_value=RandUniform(low=1.0, high=100.0),
    ... )
    >>> generator
    TruncExponentialTensorGenerator(
      (rate): RandUniformTensorGenerator(low=1.0, high=10.0)
      (max_value): RandUniformTensorGenerator(low=1.0, high=100.0)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(
        self, rate: BaseTensorGenerator | dict, max_value: BaseTensorGenerator | dict
    ) -> None:
        super().__init__()
        self._rate = setup_tensor_generator(rate)
        self._max_value = setup_tensor_generator(max_value)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"rate": self._rate, "max_value": self._max_value}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return trunc_exponential(
            rate=self._rate.generate(size=size, rng=rng),
            max_value=self._max_value.generate(size=size, rng=rng),
            generator=rng,
        )
