r"""Contain transformers to select or filter keys in the dictionary of
data."""

from __future__ import annotations

__all__ = ["SelectKeysTransformer"]

import logging
from typing import TYPE_CHECKING

from coola.utils import repr_indent, repr_mapping

from startorch.transformer.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import torch

logger = logging.getLogger(__name__)


class SelectKeysTransformer(BaseTransformer):
    r"""Implements a transformer that selects a subset of keys.

    Args:
        keys: The keys to select. The other keys will not be in the
            output.
        missing_ok: If ``False``, an exception is raised if the key is
            missing. Otherwise, the missing key is ignored.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import SelectKeysTransformer
    >>> transformer = SelectKeysTransformer(keys=["input1", "input2"])
    >>> transformer
    SelectKeysTransformer(
      (keys): ('input1', 'input2')
      (missing_ok): False
    )
    >>> data = {
    ...     "input1": torch.tensor([[0.0, -1.0, 2.0], [-4.0, 5.0, -6.0]]),
    ...     "input2": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ...     "input3": torch.randn(2, 4),
    ... }
    >>> out = transformer.transform(data)
    >>> out
    {'input1': tensor([[ 0., -1.,  2.], [-4.,  5., -6.]]),
     'input2': tensor([[1., 2., 3.], [4., 5., 6.]])}

    ```
    """

    def __init__(self, keys: Sequence[str], missing_ok: bool = False) -> None:
        self._keys = tuple(keys)
        self._missing_ok = missing_ok

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"keys": self._keys, "missing_ok": self._missing_ok}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def transform(
        self,
        data: dict[Hashable, torch.Tensor],
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> dict[Hashable, torch.Tensor]:
        out = {}
        for key in self._keys:
            val = data.get(key)
            if val is None:
                if not self._missing_ok:
                    msg = f"{key} is missing. Please use missing_ok=True if the key can be missing"
                    raise KeyError(msg)
            else:
                out[key] = val
        return out
