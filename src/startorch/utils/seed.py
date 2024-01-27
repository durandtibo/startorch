r"""Contain utility functions to manage random seeds."""

from __future__ import annotations

__all__ = ["get_random_seed", "get_torch_generator", "setup_torch_generator"]


import torch


def get_random_seed(seed: int) -> int:
    r"""Get a random seed.

    Args:
        seed: Specifies a random seed to make the process
            reproducible.

    Returns:
        int: A random seed. The value is between ``-2 ** 63`` and
            ``2 ** 63 - 1``.

    Example usage:

    ```pycon
    >>> from startorch.utils.seed import get_random_seed
    >>> get_random_seed(44)
    6176747449835261347

    ```
    """
    return torch.randint(-(2**63), 2**63 - 1, size=(1,), generator=get_torch_generator(seed)).item()


def get_torch_generator(
    random_seed: int = 1, device: torch.device | str | None = "cpu"
) -> torch.Generator:
    r"""Create a ``torch.Generator`` initialized with a given seed.

    Args:
        random_seed: Specifies a random seed.
        device: Specifies the desired device for the generator.

    Returns:
        ``torch.Generator``

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.utils.seed import get_torch_generator
    >>> generator = get_torch_generator(42)
    >>> torch.rand(2, 4, generator=generator)
    tensor([[0.8823, 0.9150, 0.3829, 0.9593],
            [0.3904, 0.6009, 0.2566, 0.7936]])
    >>> generator = get_torch_generator(42)
    >>> torch.rand(2, 4, generator=generator)
    tensor([[0.8823, 0.9150, 0.3829, 0.9593],
            [0.3904, 0.6009, 0.2566, 0.7936]])

    ```
    """
    generator = torch.Generator(device)
    generator.manual_seed(random_seed)
    return generator


def setup_torch_generator(generator_or_seed: int | torch.Generator) -> torch.Generator:
    r"""Set up a ``torch.Generator`` object.

    Args:
        generator_or_seed: Specifies a ``torch.Generator`` object or
            a random seed.

    Returns:
        A ``torch.Generator`` object.

    Example usage:

    ```pycon
    >>> from startorch.utils.seed import setup_torch_generator
    >>> generator = setup_torch_generator(42)
    >>> generator
    <torch._C.Generator object at 0x...>

    ```
    """
    if isinstance(generator_or_seed, torch.Generator):
        return generator_or_seed
    return get_torch_generator(generator_or_seed)
