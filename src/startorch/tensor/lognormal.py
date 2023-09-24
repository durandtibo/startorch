from __future__ import annotations

__all__ = [
    "LogNormalTensorGenerator",
    "RandLogNormalTensorGenerator",
    "RandTruncLogNormalTensorGenerator",
    "TruncLogNormalTensorGenerator",
]


from coola.utils.format import str_indent, str_mapping
from torch import Generator, Tensor

from startorch.random import (
    log_normal,
    rand_log_normal,
    rand_trunc_log_normal,
    trunc_log_normal,
)
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator


class LogNormalTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    log-Normal distribution.

    Args:
    ----
        mean (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) to generate the mean
            of the underlying Normal distribution.
        std (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) to generate the standard
            deviation of the underlying Normal distribution.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import LogNormal, RandUniform
        >>> generator = LogNormal(
        ...     mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0)
        ... )
        >>> generator
        LogNormalTensorGenerator(
          (mean): RandUniformTensorGenerator(low=-1.0, high=1.0)
          (std): RandUniformTensorGenerator(low=1.0, high=2.0)
        )
        >>> generator.generate(size=(2, 6))
        tensor([[...]])
    """

    def __init__(self, mean: BaseTensorGenerator | dict, std: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._mean = setup_tensor_generator(mean)
        self._std = setup_tensor_generator(std)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"mean": self._mean, "std": self._std}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return log_normal(
            mean=self._mean.generate(size=size, rng=rng),
            std=self._std.generate(size=size, rng=rng),
            generator=rng,
        )


class RandLogNormalTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    log-Normal distribution.

    Args:
    ----
        mean (float, optional): Specifies the mean of the underlying
            Normal distribution. Default: ``0.0``
        std (float, optional): Specifies the standard deviation of the
            underlying Normal distribution. Default: ``1.0``

    Raises:
    ------
        ValueError if ``std`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandLogNormal
        >>> generator = RandLogNormal(mean=0.0, std=1.0)
        >>> generator
        RandLogNormalTensorGenerator(mean=0.0, std=1.0)
        >>> generator.generate(size=(2, 6))
        tensor([[...]])
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self._mean = float(mean)
        if std <= 0:
            raise ValueError(f"std has to be greater than 0 (received: {std})")
        self._std = float(std)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(mean={self._mean}, std={self._std})"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return rand_log_normal(
            size=size,
            mean=self._mean,
            std=self._std,
            generator=rng,
        )


class RandTruncLogNormalTensorGenerator(BaseTensorGenerator):
    r"""Implements a tensor generator to generate cyclic tensors by
    sampling values from a truncated log-Normal distribution.

    Args:
    ----
        mean (float, optional): Specifies the mean of the
            log-Normal distribution. Default: ``0.0``
        std (float, optional): Specifies the standard deviation of
            the log-Normal distribution. Default: ``1.0``
        min_value (float, optional): Specifies the minimum value.
            Default: ``0.0``
        max_value (float, optional): Specifies the maximum value.
            Default: ``5.0``

    Raises:
    ------
        ValueError if ``std`` is not a positive number.
        ValueError if ``max_value`` is lower than ``min_value``.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandTruncLogNormal
        >>> generator = RandTruncLogNormal(mean=0.0, std=1.0, min_value=0.0, max_value=1.0)
        >>> generator
        RandTruncLogNormalTensorGenerator(mean=0.0, std=1.0, min_value=0.0, max_value=1.0)
        >>> generator.generate(size=(2, 6))
        tensor([[...]])
    """

    def __init__(
        self, mean: float = 0.0, std: float = 1.0, min_value: float = 0.0, max_value: float = 5.0
    ) -> None:
        super().__init__()
        self._mean = float(mean)
        if std <= 0:
            raise ValueError(f"std has to be greater than 0 (received: {std})")
        self._std = float(std)
        if max_value < min_value:
            raise ValueError(
                f"max_value ({max_value}) has to be greater or equal to min_value ({min_value})"
            )
        self._min_value = float(min_value)
        self._max_value = float(max_value)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(mean={self._mean}, std={self._std}, "
            f"min_value={self._min_value}, max_value={self._max_value})"
        )

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return rand_trunc_log_normal(
            size=size,
            mean=self._mean,
            std=self._std,
            min_value=self._min_value,
            max_value=self._max_value,
            generator=rng,
        )


class TruncLogNormalTensorGenerator(BaseTensorGenerator):
    r"""Implements a class to generate tensor by sampling values from a
    truncated log-Normal distribution.

    Args:
    ----
        mean (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) to generate the mean of
            the underlying Normal distribution.
        std (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) to generate the standard
            deviation of the underlying Normal distribution.
        min_value (``BaseTensorGenerator`` or dict): Specifies a
            tensor generator (or its configuration) to generate the
            minimum value (included).
        max_value (``BaseTensorGenerator`` or dict): Specifies a
            tensor generator (or its configuration) to generate the
            maximum value (excluded).

    Example usage:

    .. code-block:: pycon

        >>> from startorch.tensor import RandUniform, TruncLogNormal
        >>> generator = TruncLogNormal(
        ...     mean=RandUniform(low=-1.0, high=1.0),
        ...     std=RandUniform(low=1.0, high=2.0),
        ...     min_value=RandUniform(low=0.0, high=2.0),
        ...     max_value=RandUniform(low=5.0, high=10.0),
        ... )
        >>> generator
        TruncLogNormalTensorGenerator(
          (mean): RandUniformTensorGenerator(low=-1.0, high=1.0)
          (std): RandUniformTensorGenerator(low=1.0, high=2.0)
          (min_value): RandUniformTensorGenerator(low=0.0, high=2.0)
          (max_value): RandUniformTensorGenerator(low=5.0, high=10.0)
        )
        >>> generator.generate(size=(2, 6))
        tensor([[...]])
    """

    def __init__(
        self,
        mean: BaseTensorGenerator | dict,
        std: BaseTensorGenerator | dict,
        min_value: BaseTensorGenerator | dict,
        max_value: BaseTensorGenerator | dict,
    ) -> None:
        super().__init__()
        self._mean = setup_tensor_generator(mean)
        self._std = setup_tensor_generator(std)
        self._min_value = setup_tensor_generator(min_value)
        self._max_value = setup_tensor_generator(max_value)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "mean": self._mean,
                    "std": self._std,
                    "min_value": self._min_value,
                    "max_value": self._max_value,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: Generator | None = None) -> Tensor:
        return trunc_log_normal(
            mean=self._mean.generate(size=size, rng=rng),
            std=self._std.generate(size=size, rng=rng),
            min_value=self._min_value.generate(size=size, rng=rng),
            max_value=self._max_value.generate(size=size, rng=rng),
            generator=rng,
        )
