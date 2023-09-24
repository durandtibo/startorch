from __future__ import annotations

__all__ = [
    "HalfCauchyTensorGenerator",
    "RandHalfCauchyTensorGenerator",
    "RandTruncHalfCauchyTensorGenerator",
    "TruncHalfCauchyTensorGenerator",
]


from coola.utils.format import str_indent, str_mapping
from torch import Generator, Tensor

from startorch.random import (
    half_cauchy,
    rand_half_cauchy,
    rand_trunc_half_cauchy,
    trunc_half_cauchy,
)
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator


class HalfCauchyTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    half-Cauchy distribution.

    Args:
    ----
        scale (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) to generate the scale.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import HalfCauchy, RandUniform
        >>> generator = HalfCauchy(scale=RandUniform(low=1.0, high=2.0))
        >>> generator
        HalfCauchyTensorGenerator(
          (scale): RandUniformTensorGenerator(low=1.0, high=2.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(self, scale: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._scale = setup_tensor_generator(scale)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"scale": self._scale}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return half_cauchy(
            scale=self._scale.generate(size=size, rng=rng),
            generator=rng,
        )


class RandHalfCauchyTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    half-Cauchy distribution.

    Args:
    ----
        scale (float, optional): Specifies the scale of the
            distribution. Default: ``1.0``

    Raises:
    ------
        ValueError if ``scale`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandHalfCauchy
        >>> generator = RandHalfCauchy(scale=1.0)
        >>> generator
        RandHalfCauchyTensorGenerator(scale=1.0)
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale has to be greater than 0 (received: {scale})")
        self._scale = float(scale)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(scale={self._scale})"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return rand_half_cauchy(
            size=size,
            scale=self._scale,
            generator=rng,
        )


class RandTruncHalfCauchyTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    truncated half-Cauchy distribution.

    Args:
    ----
        scale (float, optional): Specifies the scale of the
            distribution. Default: ``1.0``
        max_value (float, optional): Specifies the maximum value.
            Default: ``4.0``

    Raises:
    ------
        ValueError if ``scale`` is not a positive number.
        ValueError if ``max_value`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandTruncHalfCauchy
        >>> generator = RandTruncHalfCauchy(scale=1.0, max_value=5.0)
        >>> generator
        RandTruncHalfCauchyTensorGenerator(scale=1.0, max_value=5.0)
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(
        self,
        scale: float = 1.0,
        max_value: float = 4.0,
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale has to be greater than 0 (received: {scale})")
        self._scale = float(scale)
        if max_value <= 0:
            raise ValueError(f"max_value has to be greater than 0 (received: {max_value})")
        self._max_value = float(max_value)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(scale={self._scale}, max_value={self._max_value})"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return rand_trunc_half_cauchy(
            size=size,
            scale=self._scale,
            max_value=self._max_value,
            generator=rng,
        )


class TruncHalfCauchyTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    half-Cauchy distribution.

    Args:
    ----
        scale (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) to generate the scale.
        max_value (``BaseTensorGenerator`` or dict): Specifies a
            tensor generator (or its configuration) to generate the
            maximum value (excluded).

    Example usage:

    .. code-block:: pycon

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

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return trunc_half_cauchy(
            scale=self._scale.generate(size=size, rng=rng),
            max_value=self._max_value.generate(size=size, rng=rng),
            generator=rng,
        )
