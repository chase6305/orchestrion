"""Opt-in retry policies for idempotent peripheral calls."""

import math
from dataclasses import dataclass
from typing import Tuple, Type

from orchestrion.utils.numeric import is_finite_real


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
            valid = is_finite_real(value) and value >= 0 and (allow_zero or value != 0)
            if not valid:
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
        if isinstance(next_attempt, bool) or not isinstance(next_attempt, int):
            raise TypeError("next_attempt must be an integer")
        if next_attempt < 2:
            raise ValueError("next_attempt must be at least 2")
        if self.delay == 0 or self.max_delay == 0:
            return 0.0
        if self.backoff == 1:
            return min(self.delay, self.max_delay)
        try:
            # Integer bases otherwise construct an arbitrarily large integer
            # before the cap can apply. Float overflow uses the bounded fallback.
            delay = self.delay * float(self.backoff) ** (next_attempt - 2)
        except (OverflowError, ValueError):
            delay = None
        if delay is None or delay == 0:
            # The exponent alone can overflow or underflow even when scaling by
            # the initial delay would produce a representable result.
            try:
                log_delay = math.log(self.delay) + (next_attempt - 2) * math.log(
                    self.backoff
                )
            except OverflowError:
                return self.max_delay if self.backoff > 1 else 0.0
            if log_delay >= math.log(self.max_delay):
                return self.max_delay
            return math.exp(log_delay)
        return min(delay, self.max_delay)
