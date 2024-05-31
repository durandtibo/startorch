r"""Contain the implementation of example generator that generates
examples and then transformed them."""

from __future__ import annotations

__all__ = ["TransformExampleGenerator"]


from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from startorch.example.base import BaseExampleGenerator, setup_example_generator
from startorch.transformer.base import BaseTransformer, setup_transformer

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch


class TransformExampleGenerator(BaseExampleGenerator):
    r"""Implement an example generator that generates examples, and then
    transformes them.

    Args:
        generator: The example generator or its configuration.
        transformer: The data transformer or its configuration.

    Example usage:

    ```pycon

    >>> from startorch.example import TransformExampleGenerator, HypercubeClassification
    >>> from startorch.transformer import TensorTransformer
    >>> from startorch.tensor.transformer import Abs
    >>> generator = TransformExampleGenerator(
    ...     generator=HypercubeClassification(num_classes=5, feature_size=6),
    ...     transformer=TensorTransformer(
    ...         transformer=Abs(), input="feature", output="feature_transformed"
    ...     ),
    ... )
    >>> generator
    TransformExampleGenerator(
      (generator): HypercubeClassificationExampleGenerator(num_classes=5, feature_size=6, noise_std=0.2)
      (transformer): TensorTransformer(
          (transformer): AbsTensorTransformer()
          (input): feature
          (output): feature_transformed
          (exist_ok): False
        )
    )
    >>> generator.generate(batch_size=10)
    {'target': tensor([...]), 'feature': tensor([[...]]), 'feature_transformed': tensor([[...]])}

    ```
    """

    def __init__(
        self,
        generator: BaseExampleGenerator | dict,
        transformer: BaseTransformer | dict,
    ) -> None:
        super().__init__()
        self._generator = setup_example_generator(generator)
        self._transformer = setup_transformer(transformer)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping({"generator": self._generator, "transformer": self._transformer})
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> dict[Hashable, torch.Tensor]:
        data = self._generator.generate(batch_size=batch_size, rng=rng)
        return self._transformer.transform(data, rng=rng)
