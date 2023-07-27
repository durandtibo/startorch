from __future__ import annotations

__all__ = ["TimeSequenceGenerator"]

from redcat import BatchedTensorSeq
from torch import Generator

from startorch.sequence.constant import ConstantSequenceGenerator
from startorch.sequence.exponential import ExponentialSequenceGenerator
from startorch.sequence.math import CumsumSequenceGenerator
from startorch.sequence.poisson import RandPoissonSequenceGenerator
from startorch.sequence.sort import SortSequenceGenerator
from startorch.sequence.uniform import RandUniformSequenceGenerator
from startorch.sequence.wrapper import BaseWrapperSequenceGenerator


class TimeSequenceGenerator(BaseWrapperSequenceGenerator):
    r"""Implements a sequence generator to generate time sequences.

    The time is represented as a float value where the unit in the
    second:

       - ``1.2`` -> ``00:00:01.200``
       - ``61.2`` -> ``00:01:01.200``
       - ``3661.2`` -> ``01:01:01.200``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch.sequence import RandUniform, Time
        >>> generator = Time(RandUniform())
        >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
        tensor([[...]], batch_dim=0, seq_dim=1)
    """

    def generate(
        self, seq_len: int, batch_size: int = 1, rng: Generator | None = None
    ) -> BatchedTensorSeq:
        return self._sequence.generate(seq_len=seq_len, batch_size=batch_size, rng=rng)

    @classmethod
    def create_exponential_constant_time_diff(cls, rate: float = 1.0) -> TimeSequenceGenerator:
        r"""Creates a time sequence generator where the time difference
        between two consecutive steps is constant and is sampled from an
        exponential distribution.

        Args:
        ----
            rate (float, optional): Specifies the rate of the
                exponential distribution. Default: ``1.0``

        Returns:
        -------
            ``TimeSequenceGenerator``: A time sequence generator where
                the time difference between two consecutive steps is
                constant and is sampled from an exponential
                distribution.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from startorch.sequence import RandUniform, Time
            >>> generator = Time.create_exponential_constant_time_diff()
            >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
            tensor([[...]], batch_dim=0, seq_dim=1)
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
        r"""Creates a time sequence generator where the time difference
        between two consecutive steps follows an exponential
        distribution.

        Args:
        ----
            rate (float, optional): Specifies the rate of the
                exponential distribution. Default: ``1.0``

        Returns:
        -------
            ``TimeSequenceGenerator``: A time sequence generator where the
                time difference between two consecutive steps follows
                an exponential distribution.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from startorch.sequence import RandUniform, Time
            >>> generator = Time.create_exponential_time_diff()
            >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
            tensor([[...]], batch_dim=0, seq_dim=1)
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
        r"""Creates a time sequence generator where the time difference
        between two consecutive steps is constant and is sampled from a
        Poisson distribution.

        Args:
        ----
            rate (float, optional): Specifies the rate of the Poisson
                distribution. Default: ``1.0``

        Returns:
        -------
            ``TimeSequenceGenerator``: A time sequence generator where the
                time difference between two consecutive steps is
                constant and is sampled from a Poisson distribution.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from startorch.sequence import RandUniform, Time
            >>> generator = Time.create_poisson_constant_time_diff()
            >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
            tensor([[...]], batch_dim=0, seq_dim=1)
        """
        return cls(
            CumsumSequenceGenerator(
                ConstantSequenceGenerator(RandPoissonSequenceGenerator(rate, feature_size=1)),
            ),
        )

    @classmethod
    def create_poisson_time_diff(cls, rate: float = 1.0) -> TimeSequenceGenerator:
        r"""Creates a time sequence generator where the time difference
        between two consecutive steps follows a Poisson distribution.

        Args:
        ----
            rate (float, optional): Specifies the rate of the
                Poisson distribution. Default: ``1.0``

        Returns:
        -------
            ``TimeSequenceGenerator``: A time sequence generator where
                the time difference between two consecutive steps
                follows a Poisson distribution.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from startorch.sequence import RandUniform, Time
            >>> generator = Time.create_poisson_time_diff()
            >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
            tensor([[...]], batch_dim=0, seq_dim=1)
        """
        return cls(CumsumSequenceGenerator(RandPoissonSequenceGenerator(rate, feature_size=1)))

    @classmethod
    def create_uniform_constant_time_diff(
        cls,
        min_time_diff: float = 0.0,
        max_time_diff: float = 1.0,
    ) -> TimeSequenceGenerator:
        r"""Creates a time sequence generator where the time difference
        between two consecutive steps is constant and is sampled from a
        uniform distribution.

        Args:
        ----
            min_time_diff (float, optional): Specifies the minimum
                time difference between two consecutive steps.
                Default: ``0.0``
            max_time_diff (float, optional): Specifies the maximum
                time difference between two consecutive steps.
                Default: ``1.0``

        Returns:
        -------
            ``TimeSequenceGenerator``: A time sequence generator where
                the time difference between two consecutive steps
                is constant and is sampled from a uniform
                distribution.

        Raises:
        ------
            ValueError if ``min_time_diff`` is lower than 0.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from startorch.sequence import RandUniform, Time
            >>> generator = Time.create_uniform_constant_time_diff()
            >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
            tensor([[...]], batch_dim=0, seq_dim=1)
        """
        if min_time_diff < 0:
            raise ValueError(
                f"min_time_diff has to be greater or equal to 0 (received: {min_time_diff})"
            )
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
        r"""Creates a time sequence generator where the time difference
        between two consecutive steps follows a uniform distribution.

        Args:
        ----
            min_time_diff (float, optional): Specifies the minimum
                time difference between two consecutive steps.
                Default: ``0.0``
            max_time_diff (float, optional): Specifies the maximum
                time difference between two consecutive steps.
                Default: ``1.0``

        Returns:
        -------
            ``TimeSequenceGenerator``: A time sequence generator where
                the time difference between two consecutive steps
                follows a uniform distribution.

        Raises:
        ------
            ValueError if ``min_time_diff`` is lower than 0.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from startorch.sequence import RandUniform, Time
            >>> generator = Time.create_uniform_time_diff()
            >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
            tensor([[...]], batch_dim=0, seq_dim=1)
        """
        if min_time_diff < 0:
            raise ValueError(
                f"min_time_diff has to be greater or equal to 0 (received: {min_time_diff})"
            )
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
        r"""Creates a time sequence generator where the time is sampled
        from a uniform distribution.

        Args:
        ----
            min_time (float, optional): Specifies the minimum time.
                Default: ``0.0``
            max_time (float, optional): Specifies the maximum time.
                Default: ``1.0``

        Raises:
        ------
            ValueError if ``min_time`` is lower than 0.

        Returns:
        -------
            ``TimeSequenceGenerator``: A time sequence generator where
                the time is sampled from a uniform distribution.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from startorch.sequence import RandUniform, Time
            >>> generator = Time.create_uniform_time()
            >>> generator.generate(seq_len=12, batch_size=4)  # doctest:+ELLIPSIS
            tensor([[...]], batch_dim=0, seq_dim=1)
        """
        if min_time < 0:
            raise ValueError(f"min_time has to be greater or equal to 0 (received: {min_time})")
        return cls(
            SortSequenceGenerator(
                RandUniformSequenceGenerator(
                    low=min_time,
                    high=max_time,
                    feature_size=1,
                ),
            ),
        )
