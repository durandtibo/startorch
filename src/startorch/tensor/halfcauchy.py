r"""Contain the implementation of tensor generators where the values are
sampled from a Half-Cauchy distribution."""

from __future__ import annotations

__all__ = [
    "HalfCauchyTensorGenerator",
    "RandHalfCauchyTensorGenerator",
    "RandTruncHalfCauchyTensorGenerator",
    "TruncHalfCauchyTensorGenerator",
]

from typing import TYPE_CHECKING

from coola.utils.format import str_indent, str_mapping

from startorch.random import (
    half_cauchy,
    rand_half_cauchy,
    rand_trunc_half_cauchy,
    trunc_half_cauchy,
)
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator

if TYPE_CHECKING:
    import torch


class HalfCauchyTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate tensor by sampling values from a
    half-Cauchy distribution.

    Args:
        scale: A tensor generator (or its configuration) to
            generate the scale.

    Example usage:

    ```pycon

    >>> from startorch.tensor import HalfCauchy, RandUniform
    >>> generator = HalfCauchy(scale=RandUniform(low=1.0, high=2.0))
    >>> generator
    HalfCauchyTensorGenerator(
      (scale): RandUniformTensorGenerator(low=1.0, high=2.0)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, scale: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._scale = setup_tensor_generator(scale)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"scale": self._scale}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return half_cauchy(
            scale=self._scale.generate(size=size, rng=rng),
            generator=rng,
        )


class RandHalfCauchyTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate tensor by sampling values from a
    half-Cauchy distribution.

    Args:
        scale: The scale of the distribution.

    Raises:
        ValueError: if ``scale`` is not a positive number.

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandHalfCauchy
    >>> generator = RandHalfCauchy(scale=1.0)
    >>> generator
    RandHalfCauchyTensorGenerator(scale=1.0)
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        if scale <= 0:
            msg = f"scale has to be greater than 0 (received: {scale})"
            raise ValueError(msg)
        self._scale = float(scale)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(scale={self._scale})"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return rand_half_cauchy(
            size=size,
            scale=self._scale,
            generator=rng,
        )


class RandTruncHalfCauchyTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate tensor by sampling values from a
    truncated half-Cauchy distribution.

    Args:
        scale: The scale of the distribution.
        max_value: The maximum value.

    Raises:
        ValueError: if ``scale`` is not a positive number.
        ValueError: if ``max_value`` is not a positive number.

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandTruncHalfCauchy
    >>> generator = RandTruncHalfCauchy(scale=1.0, max_value=5.0)
    >>> generator
    RandTruncHalfCauchyTensorGenerator(scale=1.0, max_value=5.0)
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        scale: float = 1.0,
        max_value: float = 4.0,
    ) -> None:
        super().__init__()
        if scale <= 0:
            msg = f"scale has to be greater than 0 (received: {scale})"
            raise ValueError(msg)
        self._scale = float(scale)
        if max_value <= 0:
            msg = f"max_value has to be greater than 0 (received: {max_value})"
            raise ValueError(msg)
        self._max_value = float(max_value)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(scale={self._scale}, max_value={self._max_value})"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return rand_trunc_half_cauchy(
            size=size,
            scale=self._scale,
            max_value=self._max_value,
            generator=rng,
        )


class TruncHalfCauchyTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate tensor by sampling values from a
    half-Cauchy distribution.

    Args:
        scale: A tensor generator (or its configuration) to
            generate the scale.
        max_value: A tensor generator (or its configuration)
            to generate the maximum value (excluded).

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandUniform, TruncHalfCauchy
    >>> generator = TruncHalfCauchy(
    ...     scale=RandUniform(low=1.0, high=2.0),
    ...     max_value=RandUniform(low=5.0, high=10.0),
    ... )
    >>> generator
    TruncHalfCauchyTensorGenerator(
      (scale): RandUniformTensorGenerator(low=1.0, high=2.0)
      (max_value): RandUniformTensorGenerator(low=5.0, high=10.0)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        scale: BaseTensorGenerator | dict,
        max_value: BaseTensorGenerator | dict,
    ) -> None:
        super().__init__()
        self._scale = setup_tensor_generator(scale)
        self._max_value = setup_tensor_generator(max_value)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"scale": self._scale, "max_value": self._max_value}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return trunc_half_cauchy(
            scale=self._scale.generate(size=size, rng=rng),
            max_value=self._max_value.generate(size=size, rng=rng),
            generator=rng,
        )
