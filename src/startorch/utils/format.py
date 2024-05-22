r"""Contain utility functions to format strings."""

from __future__ import annotations

__all__ = ["str_target_object", "str_weighted_modules"]

from typing import TYPE_CHECKING

from coola.utils import str_indent
from objectory import OBJECT_TARGET

if TYPE_CHECKING:
    from collections.abc import Sequence


def str_target_object(config: dict) -> str:
    r"""Get a string that indicates the target object in the config.

    Args:
        config: A config using the ``object_factory``
            library. This dict is expected to have a key
            ``'_target_'`` to indicate the target object.

    Returns:
        A string with the target object.

    Example usage:

    ```pycon

    >>> from startorch.utils.format import str_target_object
    >>> str_target_object({OBJECT_TARGET: "something.MyClass"})
    [_target_: something.MyClass]
    >>> str_target_object({})
    [_target_: N/A]

    ```
    """
    return f"[{OBJECT_TARGET}: {config.get(OBJECT_TARGET, 'N/A')}]"


def str_weighted_modules(modules: Sequence, weights: Sequence, num_spaces: int = 2) -> str:
    r"""Compute a pretty representation of a sequence of modules where
    each module is associated to a weight.

    Args:
        modules: The modules.
        weights: The weights. The ``weights`` should have
            the same length that ``modules``.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The string representation of the modules.

    Example usage:

    ```pycon

    >>> from startorch.utils.format import str_weighted_modules
    >>> print(str_weighted_modules(modules=["abc", "something\nelse"], weights=[1, 2]))
    (0) [weight=1] abc
    (1) [weight=2] something
      else

    ```
    """
    if len(modules) != len(weights):
        msg = (
            f"`modules` and `weights` must have the same length but received {len(modules):,} "
            f"and {len(weights):,}"
        )
        raise RuntimeError(msg)

    lines = []
    for i, (item, weight) in enumerate(zip(modules, weights)):
        lines.append(f"({i}) [weight={weight}] {str_indent(str(item), num_spaces=num_spaces)}")
    return "\n".join(lines)
