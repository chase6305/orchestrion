# Retry arithmetic boundaries

Retry delay calculation now uses the existing logarithmic fallback for
intermediate underflow as well as overflow. For example, a delay of `1e300`
and backoff of `1e-200` at attempt 4 yield approximately `1e-100` instead of
zero: the power alone underflows, but the scaled result is representable.
The maximum delay still applies, and an unrepresentably small final result
still returns zero.

Integer parameters too large for floating-point timing now produce a named
`ValueError` during construction instead of leaking `OverflowError` from
`math.isfinite`.

Regression tests cover scaled underflow, capping after fallback, true final
underflow, and oversized values for all three numeric configuration fields.

```bash
python -m pytest -q tests/test_retry_limits.py tests/test_task.py
```

## Retry input isolation

CallableTask now keeps a baseline command when multiple attempts are enabled,
and gives each attempt its own deep copy. A failed SDK call can mutate its copy
without changing subsequent attempts or the caller's original nested data.
When a schema is configured, each attempt's actual copy is validated before
device invocation. Copying and validation failures are outside the retry handler.

Stop is checked both before preparation and after copying, so a stop received
during a slow copy prevents the device call. Single-attempt tasks retain their
existing copying behavior. Retried commands must support deep copying, and
copy/validation cost scales with payload size and number of attempts.
