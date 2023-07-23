from __future__ import annotations

__all__ = ["str_target_object"]

from objectory import OBJECT_TARGET


def str_target_object(config: dict) -> str:
    r"""Gets a string that indicates the target object in the config.

    Args:
    ----
        config (dict): Specifies a config using the ``object_factory``
            library. This dict is expected to have a key
            ``'_target_'`` to indicate the target object.

    Returns:
    -------
        str: A string with the target object.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.utils.format import str_target_object
        >>> str_target_object({OBJECT_TARGET: "something.MyClass"})
        [_target_: something.MyClass]
        >>> str_target_object({})
        [_target_: N/A]
    """
    return f"[{OBJECT_TARGET}: {config.get(OBJECT_TARGET, 'N/A')}]"
