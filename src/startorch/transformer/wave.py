r"""Contain transformers that compute wave-like transformations."""

from __future__ import annotations

__all__ = ["SineWaveTransformer", "sine_wave"]

import math
from typing import TYPE_CHECKING

from startorch.transformer.base import BaseTransformer
from startorch.transformer.utils import add_item, check_input_keys

if TYPE_CHECKING:
    from collections.abc import Hashable

    import torch


class SineWaveTransformer(BaseTransformer):
    r"""Implement a transformer that computes a linear transformation.

    Args:
        value: The key that contains the input values. The sine wave
            transformation is applied on these values.
        frequency: The key that contains the frequency values.
        phase: The key that contains the phase values.
        amplitude: The key that contains the amplitude values.
        output: The key that contains the output values.
        exist_ok: If ``False``, an exception is raised if the output
            key already exists. Otherwise, the value associated to the
            output key is updated.

    Example usage:

    ```pycon

    >>> import torch
    >>> from startorch.transformer import SineWave
    >>> transformer = SineWave(
    ...     value="value",
    ...     frequency="frequency",
    ...     phase="phase",
    ...     amplitude="amplitude",
    ...     output="output",
    ... )
    >>> transformer
    SineWaveTransformer(value=value, frequency=frequency, phase=phase, amplitude=amplitude, output=output, exist_ok=False)
    >>> data = {
    ...     "value": torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]),
    ...     "frequency": torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]),
    ...     "phase": torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
    ...     "amplitude": torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
    ... }
    >>> out = transformer.transform(data)
    >>> out
    {'value': tensor([[0.1000, 0.2000, 0.3000], [0.5000, 0.6000, 0.7000]]),
     'frequency': tensor([[2., 2., 2.], [4., 4., 4.]]),
     'phase': tensor([[ 1.,  1.,  1.], [-1., -1., -1.]]),
     'amplitude': tensor([[1., 1., 1.], [2., 2., 2.]]),
     'output': tensor([[ 0.7739, -0.3632, -0.9983], [-1.6829,  1.9967, -1.5478]])}

    ```
    """

    def __init__(
        self,
        value: str,
        frequency: str,
        phase: str,
        amplitude: str,
        output: str,
        exist_ok: bool = False,
    ) -> None:
        self._value = value
        self._frequency = frequency
        self._phase = phase
        self._amplitude = amplitude
        self._output = output
        self._exist_ok = exist_ok

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(value={self._value}, frequency={self._frequency}, "
            f"phase={self._phase}, amplitude={self._amplitude}, output={self._output}, "
            f"exist_ok={self._exist_ok})"
        )

    def transform(
        self,
        data: dict[Hashable, torch.Tensor],
        *,
        rng: torch.Transformer | None = None,  # noqa: ARG002
    ) -> dict[Hashable, torch.Tensor]:
        check_input_keys(data, keys=[self._value, self._frequency, self._phase, self._amplitude])
        data = data.copy()
        add_item(
            data,
            key=self._output,
            value=sine_wave(
                value=data[self._value],
                frequency=data[self._frequency],
                phase=data[self._phase],
                amplitude=data[self._amplitude],
            ),
            exist_ok=self._exist_ok,
        )
        return data


def sine_wave(
    value: torch.Tensor,
    frequency: torch.Tensor,
    phase: torch.Tensor,
    amplitude: torch.Tensor,
) -> torch.Tensor:
    r"""Return the output of the sine-wave transformation.

    This function computes the following transformation:
    ``output = sin(2 * PI * value + phase) * amplitude``

    All the tensors must have the same shape.

    Args:
        value: The values that are used as input.
        frequency: The frequency values.
        phase: The phase values.
        amplitude: The amplitude values.

    Returns:
        The transformed values.

    Example usage:

    ```pycon

    >>> import math
    >>> import torch
    >>> from startorch.transformer import sine_wave
    >>> out = sine_wave(
    ...     value=torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 0.25, 0.5, 0.75, 1.0]]),
    ...     frequency=torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0]]),
    ...     phase=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi]]),
    ...     amplitude=torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
    ... )
    >>> out
    tensor([[ 0.0000e+00,  2.0000e+00, -1.7485e-07, -2.0000e+00,  3.4969e-07],
            [ 1.0000e+00, -8.7423e-08, -1.0000e+00,  1.7485e-07,  1.0000e+00]])

    ```
    """
    return value.mul(frequency).mul(2.0 * math.pi).add(phase).sin().mul(amplitude)
