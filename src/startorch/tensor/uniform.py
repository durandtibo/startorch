from __future__ import annotations

__all__ = ["RandUniformTensorGenerator", "UniformTensorGenerator"]


from coola.utils.format import str_indent, str_mapping
from torch import Generator, Tensor

from startorch.random import rand_uniform, uniform
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator


class RandUniformTensorGenerator(BaseTensorGenerator):
    r"""Implements a tensor generator by sampling values from a uniform
    distribution.

    Args:
    ----
        low (float, optional): Specifies the minimum value
            (inclusive). Default: ``0.0``
        high (float, optional): Specifies the maximum value
            (exclusive). Default: ``1.0``

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandUniform
        >>> generator = RandUniform(low=0, high=10)
        >>> generator.generate((2, 6))  # doctest:+ELLIPSIS
        tensor([[...]])
    """

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        super().__init__()
        self._low = float(low)
        if high < low:
            raise ValueError(f"high ({high}) has to be greater or equal to low ({low})")
        self._high = float(high)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(low={self._low}, high={self._high})"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return rand_uniform(size=size, low=self._low, high=self._high, generator=rng)


class UniformTensorGenerator(BaseTensorGenerator):
    r"""Implements a tensor generator by sampling values from a uniform
    distribution.

    Args:
    ----
        low (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) to generate the minimum
            value (inclusive).
        high (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) to generate the maximum
            value (exclusive).

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandUniform, Uniform
        >>> generator = UniformTensorGenerator(
        ...     low=RandUniform(low=0, high=2), high=RandUniform(low=8, high=10)
        ... )
        >>> generator.generate((2, 6))  # doctest:+ELLIPSIS
        tensor([[...]])
    """

    def __init__(self, low: BaseTensorGenerator | dict, high: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._low = setup_tensor_generator(low)
        self._high = setup_tensor_generator(high)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"low": self._low, "high": self._high}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return uniform(
            low=self._low.generate(size, rng=rng),
            high=self._high.generate(size, rng=rng),
            generator=rng,
        )
