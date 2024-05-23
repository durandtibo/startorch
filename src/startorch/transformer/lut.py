r"""Contain the implementation of a tensor transformer that replaces
indices by values from the lookup table."""

from __future__ import annotations

__all__ = ["LookupTableTransformer"]


from typing import TYPE_CHECKING

from startorch.transformer.base import BaseTensorTransformer

if TYPE_CHECKING:
    import torch


class LookupTableTransformer(BaseTensorTransformer):
    r"""Implement a tensor transformer that replaces indices by values
    from the lookup table.

    Args:
        weights: The weights of the lookup table.
        index: The key that contains the input indices of the lookup
            table. The tensor must be of long data type.
        output: The key that contains the output values from
            the lookup  table.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import LookupTable
    >>> transformer = LookupTable(
    ...     weights=torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), index="index", output="output"
    ... )
    >>> transformer
    LookupTableTransformer(size=6, index=index, output=output, exist_ok=False)
    >>> data = {"index": torch.tensor([[1, 2, 3], [4, 0, 2]])}
    >>> out = transformer.transform(data)
    >>> out
    {'index': tensor([[1, 2, 3], [4, 0, 2]]), 'output': tensor([[4., 3., 2.], [1., 5., 3.]])}

    ```
    """

    def __init__(
        self, weights: torch.Tensor, index: str, output: str, exist_ok: bool = False
    ) -> None:
        super().__init__(input=index, output=output, exist_ok=exist_ok)
        self._weights = weights

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(size={self._weights.shape[0]:,}, index={self._input}, "
            f"output={self._output}, exist_ok={self._exist_ok})"
        )

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return self._weights[tensor]
