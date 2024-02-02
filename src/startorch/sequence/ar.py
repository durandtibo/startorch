r"""Contain the implementation of sequence generators where the values
are sampled from an autoregressive process."""

from __future__ import annotations

__all__ = ["AutoRegressiveSequenceGenerator"]


import torch
from batchtensor.tensor import slice_along_seq
from coola.utils.format import str_indent, str_mapping

from startorch.sequence.base import BaseSequenceGenerator, setup_sequence_generator
from startorch.tensor.base import BaseTensorGenerator, setup_tensor_generator


class AutoRegressiveSequenceGenerator(BaseSequenceGenerator):
    r"""Implement a class to generate sequence by sampling values from an
    autoregressive process.

    Args:
        value: Specifies a sequence generator (or its configuration)
            used to generate the initial sequence values. These values
            are used to start the AR.
        coefficient: Specifies a sequence generator (or its
            configuration) used to generate the coefficients.
        noise: Specifies a sequence generator (or its configuration)
            used to generate the noise values.
        order: Specifies a tensor generator (or its configuration)
            used to generate the order of the AR.
        max_abs_value: Specifies the maximum absolute value.
            This argument ensures the values stay in the range
            ``[-max_abs_value, max_abs_value]``.
        warmup: Specifies the number of cycles used to
            initiate the AR. The initial value sampled do not follow
            an AR, so using warmup allows to initialize the AR so each
            value follows an AR.

    Raises:
        ValueError: if ``max_abs_value`` is not a positive number.
        ValueError: if ``warmup`` is not a positive number.

    Example usage:

    ```pycon
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
    tensor([[...]])

    ```
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
            msg = f"`max_abs_value` has to be positive but received {max_abs_value}"
            raise ValueError(msg)
        self._max_abs_value = float(max_abs_value)
        if warmup < 0:
            msg = f"warmup has to be positive or zero but received {warmup}"
            raise ValueError(msg)
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
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        order = int(self._order.generate((1,), rng=rng).item())
        if order < 1:
            msg = f"Order must be a positive integer but received {order}"
            raise RuntimeError(msg)
        x = self._value.generate(
            seq_len=seq_len + order * self._warmup, batch_size=batch_size, rng=rng
        )
        noise = self._noise.generate(
            seq_len=seq_len + order * self._warmup, batch_size=batch_size, rng=rng
        )
        coeffs = self._coefficient.generate(seq_len=order, batch_size=batch_size, rng=rng)
        for i in range(order, seq_len + order * self._warmup):
            x[:, i] = torch.fmod(
                torch.sum(coeffs * x[:, i - order : i], dim=1) + noise[:, i], self._max_abs_value
            )
        return slice_along_seq(x, order * self._warmup)
