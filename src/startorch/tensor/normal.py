r"""Contain the implementation of tensor generators where the values are
sampled from a Normal distribution."""

from __future__ import annotations

__all__ = [
    "NormalTensorGenerator",
    "RandNormalTensorGenerator",
    "RandTruncNormalTensorGenerator",
    "TruncNormalTensorGenerator",
]

from typing import TYPE_CHECKING

from coola.utils.format import str_indent, str_mapping

from startorch.random import normal, rand_normal, rand_trunc_normal, trunc_normal
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator

if TYPE_CHECKING:
    import torch


class NormalTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    Normal distribution.

    Args:
        mean: Specifies a tensor generator (or its configuration) to
            generate the mean.
        std: Specifies a tensor generator (or its configuration) to
            generate the standard deviation.

    Example usage:

    ```pycon

    >>> from startorch.tensor import Normal, RandUniform
    >>> generator = Normal(
    ...     mean=RandUniform(low=-1.0, high=1.0), std=RandUniform(low=1.0, high=2.0)
    ... )
    >>> generator
    NormalTensorGenerator(
      (mean): RandUniformTensorGenerator(low=-1.0, high=1.0)
      (std): RandUniformTensorGenerator(low=1.0, high=2.0)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, mean: BaseTensorGenerator | dict, std: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._mean = setup_tensor_generator(mean)
        self._std = setup_tensor_generator(std)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"mean": self._mean, "std": self._std}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return normal(
            mean=self._mean.generate(size=size, rng=rng),
            std=self._std.generate(size=size, rng=rng),
            generator=rng,
        )


class RandNormalTensorGenerator(BaseTensorGenerator):
    r"""Implement a sequence generator to generate cyclic sequences by
    sampling values from a Normal distribution.

    Args:
        mean: Specifies the mean of the Normal distribution.
        std: Specifies the standard deviation of the Normal
            distribution.

    Raises:
        ValueError: if ``std`` is not a postive number.

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandNormal
    >>> generator = RandNormal(mean=0.0, std=1.0)
    >>> generator
    RandNormalTensorGenerator(mean=0.0, std=1.0)
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self._mean = float(mean)
        if std <= 0:
            msg = f"std has to be greater than 0 (received: {std})"
            raise ValueError(msg)
        self._std = float(std)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(mean={self._mean}, std={self._std})"

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return rand_normal(
            size=size,
            mean=self._mean,
            std=self._std,
            generator=rng,
        )


class RandTruncNormalTensorGenerator(BaseTensorGenerator):
    r"""Implement a sequence generator to generate cyclic sequences by
    sampling values from a truncated Normal distribution.

    Args:
        mean: Specifies the mean of the Normal distribution.
        std: Specifies the standard deviation of the Normal
            distribution.
        min_value: Specifies the minimum value.
        max_value: Specifies the maximum value.

    Raises:
        ValueError: if ``std`` is not a postive number.
        ValueError: if ``max_value`` is lower than ``min_value``.

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandTruncNormal
    >>> generator = RandTruncNormal(mean=0.0, std=1.0, min_value=-1.0, max_value=1.0)
    >>> generator
    RandTruncNormalTensorGenerator(mean=0.0, std=1.0, min_value=-1.0, max_value=1.0)
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        min_value: float = -3.0,
        max_value: float = 3.0,
    ) -> None:
        super().__init__()
        self._mean = float(mean)
        if std <= 0:
            msg = f"std has to be greater than 0 (received: {std})"
            raise ValueError(msg)
        self._std = float(std)
        if max_value < min_value:
            msg = f"max_value ({max_value}) has to be greater or equal to min_value ({min_value})"
            raise ValueError(msg)
        self._min_value = float(min_value)
        self._max_value = float(max_value)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(mean={self._mean}, std={self._std}, "
            f"min_value={self._min_value}, max_value={self._max_value})"
        )

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return rand_trunc_normal(
            size=size,
            mean=self._mean,
            std=self._std,
            min_value=self._min_value,
            max_value=self._max_value,
            generator=rng,
        )


class TruncNormalTensorGenerator(BaseTensorGenerator):
    r"""Implement a class to generate sequence by sampling values from a
    truncated Normal distribution.

    Args:
        mean: Specifies a sequence generator (or its configuration) to
            generate the mean.
        std: Specifies a sequence generator (or its configuration) to
            generate the standard deviation.
        min_value: Specifies a sequence generator (or its
            configuration) to generate the minimum value (included).
        max_value: Specifies a sequence generator (or its
            configuration) to generate the maximum value (excluded).

    Example usage:

    ```pycon

    >>> from startorch.tensor import RandUniform, TruncNormal
    >>> generator = TruncNormal(
    ...     mean=RandUniform(low=-1.0, high=1.0),
    ...     std=RandUniform(low=1.0, high=2.0),
    ...     min_value=RandUniform(low=-10.0, high=-5.0),
    ...     max_value=RandUniform(low=5.0, high=10.0),
    ... )
    >>> generator
    TruncNormalTensorGenerator(
      (mean): RandUniformTensorGenerator(low=-1.0, high=1.0)
      (std): RandUniformTensorGenerator(low=1.0, high=2.0)
      (min_value): RandUniformTensorGenerator(low=-10.0, high=-5.0)
      (max_value): RandUniformTensorGenerator(low=5.0, high=10.0)
    )
    >>> generator.generate((2, 6))
    tensor([[...]])

    ```
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

    def generate(self, size: tuple[int, ...], rng: torch.Generator | None = None) -> torch.Tensor:
        return trunc_normal(
            mean=self._mean.generate(size=size, rng=rng),
            std=self._std.generate(size=size, rng=rng),
            min_value=self._min_value.generate(size=size, rng=rng),
            max_value=self._max_value.generate(size=size, rng=rng),
            generator=rng,
        )
