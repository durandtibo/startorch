from __future__ import annotations

__all__ = [
    "AcoshTensorGenerator",
    "AsinhTensorGenerator",
    "AtanhTensorGenerator",
    "CoshTensorGenerator",
    "SinhTensorGenerator",
    "TanhTensorGenerator",
]

from torch import Generator, Tensor

from startorch.tensor.wrapper import BaseWrapperTensorGenerator


class AcoshTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a sequence generator that computes the inverse
    hyperbolic cosine (arccosh) of each value.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Acosh, RandUniform
        >>> generator = Acosh(RandUniform(low=1.0, high=5.0))
        >>> generator.generate((2, 6))  # doctest:+ELLIPSIS
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._tensor.generate(size=size, rng=rng).acosh()


class AsinhTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a sequence generator that computes the inverse
    hyperbolic sine (arcsinh) of each value.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Asinh, RandUniform
        >>> generator = Asinh(RandUniform(low=0.0, high=1000.0))
        >>> generator.generate((2, 6))  # doctest:+ELLIPSIS
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._tensor.generate(size=size, rng=rng).asinh()


class AtanhTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a sequence generator that computes the inverse
    hyperbolic tangent (arctanh) of each value.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Atanh, RandUniform
        >>> generator = Atanh(RandUniform(low=-0.5, high=0.5))
        >>> generator.generate((2, 6))  # doctest:+ELLIPSIS
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._tensor.generate(size=size, rng=rng).atanh()


class CoshTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a sequence generator that computes the hyperbolic
    cosine (cosh) of each value.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Cosh, RandUniform
        >>> generator = Cosh(RandUniform())
        >>> generator.generate((2, 6))  # doctest:+ELLIPSIS
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._tensor.generate(size=size, rng=rng).cosh()


class SinhTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a sequence generator that computes the hyperbolic sine
    (sinh) of each value.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandUniform, Sinh
        >>> generator = Sinh(RandUniform(low=0.0, high=1.0))
        >>> generator.generate((2, 6))  # doctest:+ELLIPSIS
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._tensor.generate(size=size, rng=rng).sinh()


class TanhTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a sequence generator that computes the hyperbolic
    tangent (tanh) of each value.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandUniform, Tanh
        >>> generator = Tanh(RandUniform(low=0.0, high=1.0))
        >>> generator.generate((2, 6))  # doctest:+ELLIPSIS
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._tensor.generate(size=size, rng=rng).tanh()
