from __future__ import annotations

__all__ = [
    "AbsTensorGenerator",
    "AddScalarTensorGenerator",
    "AddTensorGenerator",
    "ClampTensorGenerator",
    "DivTensorGenerator",
    "ExpTensorGenerator",
    "FmodTensorGenerator",
    "LogTensorGenerator",
    "MulScalarTensorGenerator",
    "MulTensorGenerator",
    "NegTensorGenerator",
    "SqrtTensorGenerator",
    "SubTensorGenerator",
]

from collections.abc import Sequence

from coola.utils.format import str_indent, str_mapping, str_sequence
from torch import Generator, Tensor

from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator
from startorch.tensor.wrapper import BaseWrapperTensorGenerator


class AbsTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a tensor generator that computes the absolute value of
    a tensor.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Abs, RandNormal
        >>> generator = Abs(RandNormal())
        >>> generator
        AbsTensorGenerator(
          (tensor): RandNormalTensorGenerator(mean=0.0, std=1.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._generator.generate(size=size, rng=rng).abs()


class AddScalarTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a tensor generator that adds a scalar value to a
    generated batch of tensors.

    Args:
    ----
        tensor (``BaseTensorGenerator`` or dict):
            Specifies the tensor generator or its configuration.
        value (int or float): Specifies the scalar value to add.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import AddScalar, RandUniform
        >>> generator = AddScalar(RandUniform(), 10.0)
        >>> generator
        AddScalarTensorGenerator(
          (tensor): RandUniformTensorGenerator(low=0.0, high=1.0)
          (value): 10.0
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(
        self,
        generator: BaseTensorGenerator | dict,
        value: int | float,
    ) -> None:
        super().__init__(generator=generator)
        self._value = value

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"tensor": self._generator, "value": self._value}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        tensor = self._generator.generate(size=size, rng=rng)
        tensor.add_(self._value)
        return tensor


class AddTensorGenerator(BaseTensorGenerator):
    r"""Implements a tensor generator that adds several tensor.

    ``tensor = tensor_1 + tensor_2 + ... + tensor_n``

    Args:
    ----
        tensors (``Tensor``): Specifies the tensor generators.

    Raises:
    ------
        ValueError if no sequence generator is provided.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Add, RandNormal, RandUniform
        >>> generator = Add([RandUniform(), RandNormal()])
        >>> generator
        AddTensorGenerator(
          (0): RandUniformTensorGenerator(low=0.0, high=1.0)
          (1): RandNormalTensorGenerator(mean=0.0, std=1.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(self, tensors: Sequence[BaseTensorGenerator | dict]) -> None:
        super().__init__()
        if not tensors:
            raise ValueError(
                "No tensor generator. You need to specify at least one tensor generator"
            )
        self._tensors = tuple(setup_tensor_generator(tensor) for tensor in tensors)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._tensors))}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        output = self._tensors[0].generate(size=size, rng=rng)
        for generator in self._tensors[1:]:
            output.add_(generator.generate(size=size, rng=rng))
        return output


class ClampTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a tensor generator to generate tensors where the
    values are clamped.

    Note: ``min_value`` and ``max_value`` cannot be both ``None``.

    Args:
    ----
        tensor (``BaseTensorGenerator`` or dict):
            Specifies the tensor generator or its configuration.
        min_value (int, float or ``None``): Specifies the lower bound.
            If ``min_value`` is ``None``, there is no lower bound.
        max_value (int, float or ``None``): Specifies the upper bound.
            If ``max_value`` is ``None``, there is no upper bound.

    Raises:
    ------
        ValueError if both ``min`` and ``max`` are ``None``

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Clamp, RandUniform
        >>> generator = Clamp(RandUniform(low=1.0, high=50.0), min_value=2.0, max_value=10.0)
        >>> generator
        ClampTensorGenerator(
          (tensor): RandUniformTensorGenerator(low=1.0, high=50.0)
          (min_value): 2.0
          (max_value): 10.0
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(
        self,
        generator: BaseTensorGenerator | dict,
        min_value: int | float | None,
        max_value: int | float | None,
    ) -> None:
        super().__init__(generator=generator)
        if min_value is None and max_value is None:
            raise ValueError("`min_value` and `max_value` cannot be both None")
        self._min_value = min_value
        self._max_value = max_value

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "tensor": self._generator,
                    "min_value": self._min_value,
                    "max_value": self._max_value,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._generator.generate(size=size, rng=rng).clamp(self._min_value, self._max_value)


class DivTensorGenerator(BaseTensorGenerator):
    r"""Implements a tensor generator that divides one tensor by another
    one.

    ``tensor = dividend / divisor`` (a.k.a. true division)

    Args:
    ----
        dividend (``Tensor``): (``BaseTensorGenerator`` or dict):
            Specifies the dividend tensor generator or its
            configuration.
        divisor (``Tensor``): (``BaseTensorGenerator`` or dict):
            Specifies the divisor tensor generator or its
            configuration.
        rounding_mode (str or ``None``, optional): Specifies the
            type of rounding applied to the result.
            - ``None``: true division.
            - ``"trunc"``: rounds the results of the division
                towards zero.
            - ``"floor"``: floor division.
            Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Div, RandUniform
        >>> generator = Div(RandUniform(), RandUniform(low=1.0, high=10.0))
        >>> generator
        DivTensorGenerator(
          (dividend): RandUniformTensorGenerator(low=0.0, high=1.0)
          (divisor): RandUniformTensorGenerator(low=1.0, high=10.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(
        self,
        dividend: BaseTensorGenerator | dict,
        divisor: BaseTensorGenerator | dict,
        rounding_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._dividend = setup_tensor_generator(dividend)
        self._divisor = setup_tensor_generator(divisor)
        self._rounding_mode = rounding_mode

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"dividend": self._dividend, "divisor": self._divisor}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._dividend.generate(size=size, rng=rng).div(
            self._divisor.generate(size=size, rng=rng),
            rounding_mode=self._rounding_mode,
        )


class ExpTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a tensor generator that computes the exponential of a
    tensor.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Exp, RandUniform
        >>> generator = Exp(RandUniform(low=1.0, high=5.0))
        >>> generator
        ExpTensorGenerator(
          (tensor): RandUniformTensorGenerator(low=1.0, high=5.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._generator.generate(size=size, rng=rng).exp()


class FmodTensorGenerator(BaseTensorGenerator):
    r"""Implements a tensor generator that computes the element-wise
    remainder of division.

    Args:
    ----
        dividend (``BaseTensorGenerator`` or dict):
            Specifies the tensor generator (or its configuration) that
            generates the dividend values.
        divisor (int or float): Specifies the divisor.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Fmod, RandUniform
        >>> generator = Fmod(dividend=RandUniform(low=-100, high=100), divisor=10.0)
        >>> generator
        FmodTensorGenerator(
          (dividend): RandUniformTensorGenerator(low=-100.0, high=100.0)
          (divisor): 10.0
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
        >>> generator = Fmod(
        ...     dividend=RandUniform(low=-100, high=100), divisor=RandUniform(low=1, high=10)
        ... )
        >>> generator
        FmodTensorGenerator(
          (dividend): RandUniformTensorGenerator(low=-100.0, high=100.0)
          (divisor): RandUniformTensorGenerator(low=1.0, high=10.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(
        self,
        dividend: BaseTensorGenerator | dict,
        divisor: BaseTensorGenerator | dict | int | float,
    ) -> None:
        super().__init__()
        self._dividend = setup_tensor_generator(dividend)
        self._divisor = setup_tensor_generator(divisor) if isinstance(divisor, dict) else divisor

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"dividend": self._dividend, "divisor": self._divisor}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        tensor = self._dividend.generate(size=size, rng=rng)
        divisor = self._divisor
        if isinstance(divisor, BaseTensorGenerator):
            divisor = divisor.generate(size=size, rng=rng)
        tensor.fmod_(divisor)
        return tensor


class LogTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a tensor generator that computes the logarithm of a
    tensor.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import Log, RandUniform
        >>> generator = Log(RandUniform(low=1.0, high=100.0))
        >>> generator
        LogTensorGenerator(
          (tensor): RandUniformTensorGenerator(low=1.0, high=100.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._generator.generate(size=size, rng=rng).log()


class MulTensorGenerator(BaseTensorGenerator):
    r"""Implements a tensor generator that multiplies multiple tensors.

    ``tensor = tensor_1 * tensor_2 * ... * tensor_n``

    Args:
    ----
        generators (``Sequence``): Specifies the tensor generators.

    Raises:
    ------
        ValueError if no sequence generator is provided.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.tensor import Mul, RandUniform, RandNormal
        >>> generator = Mul([RandUniform(), RandNormal()])
        >>> generator
        MulTensorGenerator(
          (0): RandUniformTensorGenerator(low=0.0, high=1.0)
          (1): RandNormalTensorGenerator(mean=0.0, std=1.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(self, generators: Sequence[BaseTensorGenerator | dict]) -> None:
        super().__init__()
        if not generators:
            raise ValueError(
                "No tensor generator. You need to specify at least one tensor generator"
            )
        self._tensors = tuple(setup_tensor_generator(generator) for generator in generators)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._tensors))}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        output = self._tensors[0].generate(size=size, rng=rng)
        for generator in self._tensors[1:]:
            output.mul_(generator.generate(size=size, rng=rng))
        return output


class MulScalarTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a tensor generator that multiplies a scalar value to a
    generated batch of tensors.

    Args:
    ----
        tensor (``BaseTensorGenerator`` or dict):
            Specifies the tensor generator or its configuration.
        value (int or float): Specifies the scalar value to multiply.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.tensor import MulScalar, RandUniform, RandNormal
        >>> generator = MulScalar(RandUniform(), 42)
        >>> generator
        MulScalarTensorGenerator(
          (tensor): RandUniformTensorGenerator(low=0.0, high=1.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(
        self,
        generator: BaseTensorGenerator | dict,
        value: int | float,
    ) -> None:
        super().__init__(generator=generator)
        self._value = value

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"tensor": self._generator}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        tensor = self._generator.generate(size=size, rng=rng)
        tensor.mul_(self._value)
        return tensor


class NegTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a tensor generator that computes the negation of a
    generated tensor.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.tensor import Neg, RandNormal
        >>> generator = Neg(RandNormal())
        >>> generator
        NegTensorGenerator(
          (tensor): RandNormalTensorGenerator(mean=0.0, std=1.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return -self._generator.generate(size=size, rng=rng)


class SqrtTensorGenerator(BaseWrapperTensorGenerator):
    r"""Implements a tensor generator that computes the squared root of a
    tensor.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandUniform, Sqrt
        >>> generator = Sqrt(RandUniform(low=1.0, high=100.0))
        >>> generator
        SqrtTensorGenerator(
          (tensor): RandUniformTensorGenerator(low=1.0, high=100.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._generator.generate(size=size, rng=rng).sqrt()


class SubTensorGenerator(BaseTensorGenerator):
    r"""Implements a tensor generator that subtracts two tensors.

    ``tensor = tensor_1 - tensor_2``

    Args:
    ----
        tensor1 (``BaseTensorGenerator`` or dict):
            Specifies the first tensor generator or its
            configuration.
        tensor2 (``BaseTensorGenerator`` or dict):
            Specifies the second tensor generator or its
            configuration.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandNormal, RandUniform, Sub
        >>> generator = Sub(RandUniform(), RandNormal())
        >>> generator
        SubTensorGenerator(
          (tensor1): RandUniformTensorGenerator(low=0.0, high=1.0)
          (tensor2): RandNormalTensorGenerator(mean=0.0, std=1.0)
        )
        >>> generator.generate((2, 6))
        tensor([[...]])
    """

    def __init__(
        self, tensor1: BaseTensorGenerator | dict, tensor2: BaseTensorGenerator | dict
    ) -> None:
        super().__init__()
        self._tensor1 = setup_tensor_generator(tensor1)
        self._tensor2 = setup_tensor_generator(tensor2)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"tensor1": self._tensor1, "tensor2": self._tensor2}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return self._tensor1.generate(size=size, rng=rng).sub(
            self._tensor2.generate(size=size, rng=rng)
        )
