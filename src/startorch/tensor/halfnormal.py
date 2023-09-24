from __future__ import annotations

__all__ = [
    "HalfNormalTensorGenerator",
    "RandHalfNormalTensorGenerator",
    "RandTruncHalfNormalTensorGenerator",
    "TruncHalfNormalTensorGenerator",
]


from coola.utils.format import str_indent, str_mapping
from torch import Generator, Tensor

from startorch.random import (
    half_normal,
    rand_half_normal,
    rand_trunc_half_normal,
    trunc_half_normal,
)
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator


class HalfNormalTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    half-Normal distribution.

    Args:
    ----
        std (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) to generate the standard
            deviation.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import HalfNormal, RandUniform
        >>> generator = HalfNormal(std=RandUniform(low=1.0, high=2.0))
        >>> generator
        HalfNormalTensorGenerator(
          (std): RandUniformTensorGenerator(low=1.0, high=2.0)
        )
        >>> generator.generate(size=(2, 6))
        tensor([[...]])
    """

    def __init__(self, std: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._std = setup_tensor_generator(std)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"std": self._std}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return half_normal(
            std=self._std.generate(size=size, rng=rng),
            generator=rng,
        )


class RandHalfNormalTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    half-Normal distribution.

    Args:
    ----
        std (float, optional): Specifies the std of the distribution.
            Default: ``1.0``

    Raises:
    ------
        ValueError if ``std`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandHalfNormal
        >>> generator = RandHalfNormal(std=1.0)
        >>> generator
        RandHalfNormalTensorGenerator(std=1.0)
        >>> generator.generate(size=(2, 6))
        tensor([[...]])
    """

    def __init__(self, std: float = 1.0) -> None:
        super().__init__()
        if std <= 0:
            raise ValueError(f"std has to be greater than 0 (received: {std})")
        self._std = float(std)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(std={self._std})"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return rand_half_normal(
            size=size,
            std=self._std,
            generator=rng,
        )


class RandTruncHalfNormalTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    truncated half-Normal distribution.

    Args:
    ----
        std (float, optional): Specifies the std of the distribution.
            Default: ``1.0``
        max_value (float, optional): Specifies the maximum value.
            Default: ``3.0``

    Raises:
    ------
        ValueError if ``std`` is not a positive number.
        ValueError if ``max_value`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandTruncHalfNormal
        >>> generator = RandTruncHalfNormal(std=1.0, max_value=1.0)
        >>> generator
        RandTruncHalfNormalTensorGenerator(std=1.0, max_value=1.0)
        >>> generator.generate(size=(2, 6))
        tensor([[...]])
    """

    def __init__(self, std: float = 1.0, max_value: float = 3.0) -> None:
        super().__init__()
        if std <= 0:
            raise ValueError(f"std has to be greater than 0 (received: {std})")
        self._std = float(std)
        if max_value <= 0:
            raise ValueError(f"max_value has to be greater than 0 (received: {max_value})")
        self._max_value = float(max_value)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(std={self._std}, max_value={self._max_value})"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return rand_trunc_half_normal(
            size=size,
            std=self._std,
            max_value=self._max_value,
            generator=rng,
        )


class TruncHalfNormalTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    half-Normal distribution.

    Args:
    ----
        std (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) to generate the std.
        max_value (``BaseTensorGenerator`` or dict): Specifies a
            tensor generator (or its configuration) to generate the
            maximum value (excluded).

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandUniform, TruncHalfNormal
        >>> generator = TruncHalfNormal(
        ...     std=RandUniform(low=1.0, high=2.0),
        ...     max_value=RandUniform(low=5.0, high=10.0),
        ... )
        >>> generator
        TruncHalfNormalTensorGenerator(
          (std): RandUniformTensorGenerator(low=1.0, high=2.0)
          (max_value): RandUniformTensorGenerator(low=5.0, high=10.0)
        )
        >>> generator.generate(size=(2, 6))
        tensor([[...]])
    """

    def __init__(
        self,
        std: BaseTensorGenerator | dict,
        max_value: BaseTensorGenerator | dict,
    ) -> None:
        super().__init__()
        self._std = setup_tensor_generator(std)
        self._max_value = setup_tensor_generator(max_value)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"std": self._std, "max_value": self._max_value}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return trunc_half_normal(
            std=self._std.generate(size=size, rng=rng),
            max_value=self._max_value.generate(size=size, rng=rng),
            generator=rng,
        )
