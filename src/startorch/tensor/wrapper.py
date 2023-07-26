from __future__ import annotations

__all__ = ["BaseWrapperTensorGenerator"]


from coola.utils.format import str_indent, str_mapping

from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator


class BaseWrapperTensorGenerator(BaseTensorGenerator):
    r"""Defines a base class to easily wrap a tensor generator.

    Args:
    ----
        tensor (``BaseTensorGenerator`` or dict):
            Specifies the tensor generator or its configuration.
    """

    def __init__(self, tensor: BaseTensorGenerator | dict) -> None:
        super().__init__()
        self._tensor = setup_tensor_generator(tensor)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"tensor": self._tensor}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
