"""Opt-in retry policies for idempotent peripheral calls."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Tuple, Type


@dataclass(frozen=True)
class RetryPolicy:
    """Describe bounded exponential retry behavior for a callable task."""

    max_attempts: int = 1
    delay: float = 0.0
    backoff: float = 1.0
    max_delay: float = 60.0
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise TypeError("max_attempts must be an integer")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        for name, value, allow_zero in (
            ("delay", self.delay, True),
            ("backoff", self.backoff, False),
            ("max_delay", self.max_delay, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
                or (not allow_zero and value == 0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError("{} must be {} and finite".format(name, qualifier))
        if not isinstance(self.retry_exceptions, tuple) or not all(
            isinstance(exc, type) and issubclass(exc, Exception)
            for exc in self.retry_exceptions
        ):
            raise TypeError("retry_exceptions must be a tuple of exception types")
        if not self.retry_exceptions:
            raise ValueError("retry_exceptions must not be empty")

    def delay_before(self, next_attempt: int) -> float:
        """Return delay before a one-based attempt number greater than one."""
        try:
            delay = self.delay * self.backoff ** (next_attempt - 2)
        except OverflowError:
            return self.max_delay
        return min(delay, self.max_delay)
