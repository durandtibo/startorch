from __future__ import annotations

__all__ = ["AutoRegressiveSequenceGenerator"]


import torch
from coola.utils.format import str_indent, str_mapping
from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator


class AutoRegressiveSequenceGenerator(BaseSequenceGenerator):
    r"""Implements a class to generate sequence by sampling values with a
    linear pattern.

    Args:
    ----
        value (``BaseSequenceGenerator`` or dict): Specifies a
            sequence generator (or its configuration) used to generate
            the initial sequence values. These values are used to
            start the AR.
        coefficient (``BaseSequenceGenerator`` or dict): Specifies a
            sequence generator (or its configuration) used to generate
            the coefficients.
        noise (``BaseSequenceGenerator`` or dict): Specifies a sequence
            generator (or its configuration) used to generate the
            noise values.
        order (``BaseTensorGenerator`` or dict): Specifies a tensor
            generator (or its configuration) used to generate the order
            of the AR.
        max_abs_value (float): Specifies the maximum absolute value.
            This argument ensures the values stay in the range
            ``[-max_abs_value, max_abs_value]``.
        warmup (int, optional): Specifies the number of cycles used to
            initiate the AR. The initial value sampled do not follow
            an AR, so using warmup allows to initialize the AR so each
            value follows an AR. Default: ``10``

    Raises:
    ------
        ValueError if ``max_abs_value`` is not a positive number.
        ValueError if ``warmup`` is not a positive number.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import AutoRegressive, RandUniform, RandNormal, Full
        >>> from startorch.tensor import RandInt
        >>> generator = AutoRegressive(
        ...     value=RandNormal(),
        ...     coefficient=RandUniform(low=-1.0, high=1.0),
        ...     noise=Full(0.0),
        ...     order=RandInt(low=1, high=6),
        ...     max_abs_value=100.0,
        ... )
        >>> generator
        AutoRegressiveSequenceGenerator(
          (value): RandNormalSequenceGenerator(mean=0.0, std=1.0, feature_size=(1,))
          (coefficient): RandUniformSequenceGenerator(low=-1.0, high=1.0, feature_size=(1,))
          (noise): FullSequenceGenerator(value=0.0, feature_size=(1,))
          (order): RandIntTensorGenerator(low=1, high=6)
          (max_abs_value): 100.0
          (warmup): 10
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def __init__(
        self,
        value: BaseSequenceGenerator | dict,
        coefficient: BaseSequenceGenerator | dict,
        noise: BaseSequenceGenerator | dict,
        order: BaseTensorGenerator | dict,
        max_abs_value: float,
        warmup: int = 10,
    ) -> None:
        super().__init__()
        self._value = setup_sequence_generator(value)
        self._coefficient = setup_sequence_generator(coefficient)
        self._noise = setup_sequence_generator(noise)
        self._order = setup_tensor_generator(order)

        if max_abs_value <= 0.0:
            raise ValueError(f"`max_abs_value` has to be positive but received {max_abs_value}")
        self._max_abs_value = float(max_abs_value)
        if warmup < 0:
            raise ValueError(f"warmup has to be positive or zero but received {warmup}")
        self._warmup = int(warmup)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "value": self._value,
                    "coefficient": self._coefficient,
                    "noise": self._noise,
                    "order": self._order,
                    "max_abs_value": self._max_abs_value,
                    "warmup": self._warmup,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        order = int(self._order.generate((1,), rng=rng).item())
        if order < 1:
            raise RuntimeError(f"Order must be a positive integer but received {order}")
        x = self._value.generate(
            seq_len=seq_len + order * self._warmup, batch_size=batch_size, rng=rng
        ).data
        noise = self._noise.generate(
            seq_len=seq_len + order * self._warmup, batch_size=batch_size, rng=rng
        ).data
        coeffs = self._coefficient.generate(seq_len=order, batch_size=batch_size, rng=rng).data
        for i in range(order, seq_len + order * self._warmup):
            x[:, i] = torch.fmod(
                torch.sum(coeffs * x[:, i - order : i], dim=1) + noise[:, i], self._max_abs_value
            )
        return BatchedTensorSeq(x).slice_along_seq(order * self._warmup)
