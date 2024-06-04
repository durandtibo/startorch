r"""Contain the implementation of transformer that computes a linear
transformation."""

from __future__ import annotations

__all__ = ["LinearTransformer"]

from typing import TYPE_CHECKING

from startorch.transformer.base import BaseTransformer
from startorch.transformer.utils import add_item, check_input_keys

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch


class LinearTransformer(BaseTransformer):
    r"""Implement a tensor transformer that computes a linear
    transformation.

    Args:
        value: The key that contains the input values. The linear
            transformation is applied on these values.
        slope: The key that contains the slope values.
        intercept: The key that contains the intercept values.
        output: The key that contains the output values.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import Linear
    >>> transformer = Linear(
    ...     value="value", slope="slope", intercept="intercept", output="output"
    ... )
    >>> transformer
    LinearTransformer(value=value, slope=slope, intercept=intercept, output=output, exist_ok=False)
    >>> data = {
    ...     "value": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ...     "slope": torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
    ...     "intercept": torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
    ... }
    >>> out = transformer.transform(data)
    >>> out
    {'value': tensor([[1., 2., 3.], [4., 5., 6.]]),
     'slope': tensor([[2., 2., 2.], [4., 4., 4.]]),
     'intercept': tensor([[ 1.,  1.,  1.], [-1., -1., -1.]]),
     'output': tensor([[ 3.,  5.,  7.], [15., 19., 23.]])}

    ```
    """

    def __init__(
        self, value: str, slope: str, intercept: str, output: str, exist_ok: bool = False
    ) -> None:
        self._value = value
        self._slope = slope
        self._intercept = intercept
        self._output = output
        self._exist_ok = exist_ok

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(value={self._value}, slope={self._slope}, "
            f"intercept={self._intercept}, output={self._output}, exist_ok={self._exist_ok})"
        )

    def transform(
        self,
        data: dict[Hashable, torch.Tensor],
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> dict[Hashable, torch.Tensor]:
        check_input_keys(data, keys=[self._value, self._slope, self._intercept])
        data = data.copy()
        add_item(
            data,
            key=self._output,
            value=data[self._value].mul(data[self._slope]).add(data[self._intercept]),
            exist_ok=self._exist_ok,
        )
        return data
