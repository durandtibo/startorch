r"""Contain the implementation of a base class to easily wrap a sequence
generator into another sequence generator."""

from __future__ import annotations

__all__ = ["BaseWrapperSequenceGenerator"]


from coola.utils import str_indent, str_mapping

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator


class BaseWrapperSequenceGenerator(BaseSequenceGenerator):
    r"""Define a base class to easily wrap a sequence generator.

    Note:
        It is possible to wrap a sequence generator into another
        sequence generator without using this base class. This class
        makes it more convenient and reduce duplicate code.

    Args:
        generator: The sequence generator or its
            configuration.
    """

    def __init__(self, generator: BaseSequenceGenerator | dict) -> None:
        super().__init__()
        self._generator = setup_sequence_generator(generator)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"sequence": self._generator}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
