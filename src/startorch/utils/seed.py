from __future__ import annotations

__all__ = ["get_random_seed", "get_torch_generator"]


import torch


def get_random_seed(seed: int) -> int:
    r"""Gets a random seed.

    Args:
    ----
        seed (int): Specifies a random seed to make the process
            reproducible.

    Returns:
    -------
        int: A random seed. The value is between ``-2 ** 63`` and
            ``2 ** 63 - 1``.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.utils.seed import get_random_seed
        >>> get_random_seed(44)
        6176747449835261347
    """
    return torch.randint(
        -(2**63), 2**63 - 1, size=(1,), generator=get_torch_generator(seed)
    ).item()


def get_torch_generator(
    random_seed: int = 1, device: torch.device | str | None = "cpu"
) -> torch.Generator:
    r"""Creates a ``torch.Generator`` initialized with a given seed.

    Args:
    ----
        random_seed (int, optional): Specifies a random seed.
            Default: ``1``
        device (``torch.device`` or str or ``None``, optional):
            Specifies the desired device for the generator.
            Default: ``'cpu'``

    Returns:
    -------
        ``torch.Generator``

    Example usage:

    .. code-block:: pycon

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
    """
    generator = torch.Generator(device)
    generator.manual_seed(random_seed)
    return generator
