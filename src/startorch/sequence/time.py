r"""Contain the implementations of sequence generators to generate time
sequences."""

from __future__ import annotations

__all__ = ["TimeSequenceGenerator"]

from typing import TYPE_CHECKING

from startorch.sequence.constant import ConstantSequenceGenerator
from startorch.sequence.exponential import ExponentialSequenceGenerator
from startorch.sequence.math import CumsumSequenceGenerator
from startorch.sequence.poisson import RandPoissonSequenceGenerator
from startorch.sequence.sort import SortSequenceGenerator
from startorch.sequence.uniform import RandUniformSequenceGenerator
from startorch.sequence.wrapper import BaseWrapperSequenceGenerator

if TYPE_CHECKING:
    import torch


class TimeSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implement a sequence generator to generate time sequences.

    The time is represented as a float value. The unit depends on the
    context. If the unit is the second:

       - ``1.2`` -> ``00:00:01.200``
       - ``61.2`` -> ``00:01:01.200``
       - ``3661.2`` -> ``01:01:01.200``

    Example usage:

    ```pycon
    >>> import torch
    >>> from startorch.sequence import RandUniform, Time
    >>> generator = Time(RandUniform())
    >>> generator
    TimeSequenceGenerator(
      (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
    )
    >>> generator.generate(seq_len=12, batch_size=4)
    tensor([[...]])

    ```
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        return self._generator.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)

    @classmethod
    def create_exponential_constant_time_diff(cls, rate: float = 1.0) -> TimeSequenceGenerator:
        r"""Create a time sequence generator where the time difference
        between two consecutive steps is constant and is sampled from an
        exponential distribution.

        Args:
            rate: Specifies the rate of the exponential distribution.

        Returns:
            A time sequence generator where the time difference
                between two consecutive steps is constant and is
                sampled from an exponential distribution.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.sequence import RandUniform, Time
        >>> generator = Time.create_exponential_constant_time_diff()
        >>> generator
        TimeSequenceGenerator(
          (sequence): CumsumSequenceGenerator(
              (sequence): ConstantSequenceGenerator(
                  (sequence): ExponentialSequenceGenerator(
                      (rate): ConstantSequenceGenerator(
                          (sequence): RandUniformSequenceGenerator(low=1.0, high=1.0, feature_size=(1,))
                        )
                    )
                )
            )
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]])

        ```
        """
        return cls(
            CumsumSequenceGenerator(
                ConstantSequenceGenerator(
                    ExponentialSequenceGenerator.create_uniform_rate(
                        min_rate=rate,
                        max_rate=rate,
                        feature_size=1,
                    )
                ),
            ),
        )

    @classmethod
    def create_exponential_time_diff(cls, rate: float = 1.0) -> TimeSequenceGenerator:
        r"""Create a time sequence generator where the time difference
        between two consecutive steps follows an exponential
        distribution.

        Args:
            rate: Specifies the rate of the exponential distribution.

        Returns:
            A time sequence generator where the time difference between
                two consecutive steps follows an exponential
                distribution.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.sequence import RandUniform, Time
        >>> generator = Time.create_exponential_time_diff()
        >>> generator
        TimeSequenceGenerator(
          (sequence): CumsumSequenceGenerator(
              (sequence): ExponentialSequenceGenerator(
                  (rate): ConstantSequenceGenerator(
                      (sequence): RandUniformSequenceGenerator(low=1.0, high=1.0, feature_size=(1,))
                    )
                )
            )
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]])

        ```
        """
        return cls(
            CumsumSequenceGenerator(
                ExponentialSequenceGenerator.create_uniform_rate(
                    min_rate=rate,
                    max_rate=rate,
                    feature_size=1,
                ),
            ),
        )

    @classmethod
    def create_poisson_constant_time_diff(cls, rate: float = 1.0) -> TimeSequenceGenerator:
        r"""Create a time sequence generator where the time difference
        between two consecutive steps is constant and is sampled from a
        Poisson distribution.

        Args:
            rate: Specifies the rate of the Poisson distribution.

        Returns:
            A time sequence generator where the time difference
                between two consecutive steps is constant and is
                sampled from a Poisson distribution.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.sequence import RandUniform, Time
        >>> generator = Time.create_poisson_constant_time_diff()
        >>> generator
        TimeSequenceGenerator(
          (sequence): CumsumSequenceGenerator(
              (sequence): ConstantSequenceGenerator(
                  (sequence): RandPoissonSequenceGenerator(rate=1.0, feature_size=(1,))
                )
            )
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]])

        ```
        """
        return cls(
            CumsumSequenceGenerator(
                ConstantSequenceGenerator(RandPoissonSequenceGenerator(rate, feature_size=1)),
            ),
        )

    @classmethod
    def create_poisson_time_diff(cls, rate: float = 1.0) -> TimeSequenceGenerator:
        r"""Create a time sequence generator where the time difference
        between two consecutive steps follows a Poisson distribution.

        Args:
            rate: Specifies the rate of the Poisson distribution.

        Returns:
            A time sequence generator where the time difference
                between two consecutive steps follows a Poisson
                distribution.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.sequence import RandUniform, Time
        >>> generator = Time.create_poisson_time_diff()
        >>> generator
        TimeSequenceGenerator(
          (sequence): CumsumSequenceGenerator(
              (sequence): RandPoissonSequenceGenerator(rate=1.0, feature_size=(1,))
            )
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]])

        ```
        """
        return cls(CumsumSequenceGenerator(RandPoissonSequenceGenerator(rate, feature_size=1)))

    @classmethod
    def create_uniform_constant_time_diff(
        cls,
        min_time_diff: float = 0.0,
        max_time_diff: float = 1.0,
    ) -> TimeSequenceGenerator:
        r"""Create a time sequence generator where the time difference
        between two consecutive steps is constant and is sampled from a
        uniform distribution.

        Args:
            min_time_diff: Specifies the minimum time difference
                between two consecutive steps.
            max_time_diff: Specifies the maximum time difference
                between two consecutive steps.

        Returns:
            A time sequence generator where the time difference
                between two consecutive steps is constant and is
                sampled from a uniform distribution.

        Raises:
            ValueError: if ``min_time_diff`` is lower than 0.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.sequence import RandUniform, Time
        >>> generator = Time.create_uniform_constant_time_diff()
        >>> generator
        TimeSequenceGenerator(
          (sequence): CumsumSequenceGenerator(
              (sequence): ConstantSequenceGenerator(
                  (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
                )
            )
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]])

        ```
        """
        if min_time_diff < 0:
            msg = f"min_time_diff has to be greater or equal to 0 (received: {min_time_diff})"
            raise ValueError(msg)
        return cls(
            CumsumSequenceGenerator(
                ConstantSequenceGenerator(
                    RandUniformSequenceGenerator(
                        low=min_time_diff,
                        high=max_time_diff,
                        feature_size=1,
                    )
                ),
            ),
        )

    @classmethod
    def create_uniform_time_diff(
        cls,
        min_time_diff: float = 0.0,
        max_time_diff: float = 1.0,
    ) -> TimeSequenceGenerator:
        r"""Create a time sequence generator where the time difference
        between two consecutive steps follows a uniform distribution.

        Args:
            min_time_diff: Specifies the minimum time difference
                between two consecutive steps.
            max_time_diff: Specifies the maximum time difference
                between two consecutive steps.

        Returns:
            A time sequence generator where the time difference
                between two consecutive steps follows a uniform
                distribution.

        Raises:
            ValueError: if ``min_time_diff`` is lower than 0.

        Example usage:

        ```pycon
        >>> import torch
        >>> from startorch.sequence import RandUniform, Time
        >>> generator = Time.create_uniform_time_diff()
        >>> generator
        TimeSequenceGenerator(
          (sequence): CumsumSequenceGenerator(
              (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
            )
        )
        >>> generator.generate(seq_len=12, batch_size=4)
        tensor([[...]])

        ```
        """
        if min_time_diff < 0:
            msg = f"min_time_diff has to be greater or equal to 0 (received: {min_time_diff})"
            raise ValueError(msg)
        return cls(
            CumsumSequenceGenerator(
                RandUniformSequenceGenerator(
                    low=min_time_diff,
                    high=max_time_diff,
                    feature_size=1,
                ),
            ),
        )

    @classmethod
    def create_uniform_time(
        cls,
        min_time: float = 0.0,
        max_time: float = 1.0,
    ) -> TimeSequenceGenerator:
        r"""Create a time sequence generator where the time is sampled
        from a uniform distribution.

        Args:
            min_time: Specifies the minimum time.
            max_time: Specifies the maximum time.

        Returns:
            A time sequence generator where the time is sampled from a
                uniform distribution.

        Raises:
            ValueError: if ``min_time`` is lower than 0.

        Example usage:

        ```pycon

            >>> import torch
            >>> from startorch.sequence import RandUniform, Time
            >>> generator = Time.create_uniform_time()
            >>> generator
            TimeSequenceGenerator(
              (sequence): SortSequenceGenerator(
                  (sequence): RandUniformSequenceGenerator(low=0.0, high=1.0, feature_size=(1,))
                )
            )
            >>> generator.generate(seq_len=12, batch_size=4)
            tensor([[...]])
        """
        if min_time < 0:
            msg = f"min_time has to be greater or equal to 0 (received: {min_time})"
            raise ValueError(msg)
        return cls(
            SortSequenceGenerator(
                RandUniformSequenceGenerator(
                    low=min_time,
                    high=max_time,
                    feature_size=1,
                ),
            ),
        )
