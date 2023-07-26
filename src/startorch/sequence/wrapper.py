from __future__ import annotations

__all__ = ["BaseWrapperSequenceGenerator"]


from coola.utils import str_indent, str_mapping

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator


class BaseWrapperSequenceGenerator(BaseSequenceGenerator):
    r"""Defines a base class to easily wrap a sequence generator.

    Args:
    ----
        generator (``BaseSequenceGenerator`` or dict):
            Specifies the sequence generator or its configuration.
    """

    def __init__(self, sequence: BaseSequenceGenerator | dict) -> None:
        super().__init__()
        self._sequence = setup_sequence_generator(sequence)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"sequence": self._sequence}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
