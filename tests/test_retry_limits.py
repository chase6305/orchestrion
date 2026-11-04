"""Extreme retry indices must not trigger unbounded integer exponentiation."""

import subprocess
import sys
from fractions import Fraction

import pytest

from orchestrion import RetryPolicy


def test_integer_backoff_with_extreme_attempt_finishes_promptly():
    # Isolate this regression: the original integer power can run indefinitely.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from orchestrion import RetryPolicy; "
            "assert RetryPolicy(delay=1, backoff=2, max_delay=60)"
            ".delay_before(10**400) == 60",
        ],
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backoff", [2, 3, 10])
def test_integer_and_float_backoff_agree(backoff):
    integer = RetryPolicy(delay=0.125, backoff=backoff, max_delay=100)
    floating = RetryPolicy(delay=0.125, backoff=float(backoff), max_delay=100)
    for attempt in (2, 3, 4, 10, 1000):
        assert integer.delay_before(attempt) == floating.delay_before(attempt)


@pytest.mark.parametrize("field", ["delay", "backoff", "max_delay"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(10**1000, id="huge-int"),
        pytest.param(Fraction(10**1000, 3), id="huge-fraction"),
    ],
)
def test_unrepresentable_retry_parameter_has_a_validation_error(field, value):
    with pytest.raises(ValueError, match=field):
        RetryPolicy(**{field: value})


@pytest.mark.parametrize("max_delay, expected", [(60, 1e-100), (1e-150, 1e-150)])
def test_intermediate_underflow_preserves_scaled_delay_and_cap(max_delay, expected):
    policy = RetryPolicy(delay=1e300, backoff=1e-200, max_delay=max_delay)
    assert policy.delay_before(4) == pytest.approx(expected, rel=1e-12, abs=0)


def test_unrepresentably_small_final_delay_is_zero():
    assert RetryPolicy(delay=1e-300, backoff=1e-200).delay_before(4) == 0
