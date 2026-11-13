"""Numeric predicates shared by runtime timing and motion validation."""

import math
from numbers import Real


def is_finite_real(value: object) -> bool:
    """Return whether a non-boolean real has a finite float representation."""
    # Native types precede the ABC to avoid a lookup for each joint sample.
    if isinstance(value, bool) or not isinstance(value, (int, float, Real)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def is_positive_finite_real(value: object) -> bool:
    """Return whether a positive real stays positive as a finite float."""
    # A positive Fraction can underflow to zero before a native wait starts.
    return is_finite_real(value) and value > 0 and float(value) > 0
