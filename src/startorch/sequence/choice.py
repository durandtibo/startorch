from __future__ import annotations

__all__ = ["MultinomialChoiceSequenceGenerator"]

from collections.abc import Sequence

import torch
from coola.utils.format import str_indent
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.utils.format import str_weighted_modules
from startorch.utils.weight import prepare_weighted_generators


class MultinomialChoiceSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a sequence generator that select a sequence generator
    at each batch.

    This sequence generator is used to generate sequences with different
    generation processes. The user can specify a list of sequence
    generators with an associated weight. The weight is used to sample
    the sequence generator with a multinomial distribution. Higher
    weight means that the sequence generator has a higher probability
    to be selected at each batch. Each dictionary in the
    ``generators`` input should have the following items:

        - a key ``'generator'`` which indicates the sequence generator
            or its configuration.
        - an optional key ``'weight'`` with a float value which
            indicates the weight of the sequence generator.
            If this key is absent, the weight is set to ``1.0``.

    Args:
    ----
        sequences (sequence): Specifies the sequence
            generators and their weights. See above to learn
            about the expected format.

    Example usage:

    .. code-block:: pycon

        >>> from startorch.sequence import MultinomialChoice, RandUniform, RandNormal
        >>> generator = MultinomialChoice(
        ...     (
        ...         {"weight": 2.0, "generator": RandUniform()},
        ...         {"weight": 1.0, "generator": RandNormal()},
        ...     )
        ... )
        >>> generator.generate(seq_len=10, batch_size=2)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(self, sequences: Sequence[dict[str, BaseSequenceGenerator | dict]]) -> None:
        super().__init__()
        sequences, weights = prepare_weighted_generators(sequences)
        self._sequences = tuple(setup_sequence_generator(generator) for generator in sequences)
        self._weights = torch.as_tensor(weights, dtype=torch.float)

    def __repr__(self) -> str:
        args = str_indent(str_weighted_modules(modules=self._sequences, weights=self._weights))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        index = torch.multinomial(self._weights, num_samples=1, generator=rng).item()
        return self._sequences[index].generate(seq_len=seq_len, batch_size=batch_size, rng=rng)
