"""Numeric predicates shared by runtime timing and motion validation."""

import math
from numbers import Real


def is_finite_real(value: object) -> bool:
    """Return whether a non-boolean real has a finite float representation."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
