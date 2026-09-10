# Task storage and response isolation

Request deduplication now packs IDs into 256-bit blocks. Fully accepted blocks
at the beginning of the ID space are discarded and represented by a watermark.
This retains exact gap detection after result eviction, including out-of-order
submissions and arbitrarily large nonnegative IDs. Both function task adapters
share the implementation.

Previously, missing ID 0 prevented all later IDs from being compressed. Limiting
retained results therefore did not limit the per-ID history allocation. The new
representation reduces this overhead for nearby IDs; widely separated IDs still
require separate blocks, so history is not universally bounded.

Local Python 3.10 microbenchmark: 30,000 immediate threaded-task submissions,
`max_result_count=1`, median of three timing runs. Allocation was measured in a
separate run with `tracemalloc`, started after task initialization. The baseline
is the working implementation immediately before this change, not repository HEAD.

| ID sequence | Before retained bytes | After retained bytes | Before seconds | After seconds |
| --- | ---: | ---: | ---: | ---: |
| 0, 1, 2, … | 1,460 | 1,492 | 0.0739 | 0.0834 |
| 1, 2, 3, … | 2,931,416 | 13,312 | 0.0754 | 0.0838 |
| 0, 2, 4, … | 2,934,972 | 24,956 | 0.0744 | 0.0839 |

These measurements exclude device and network latency. Bitmap bookkeeping costs
about 10–13% in this no-op submission benchmark, in exchange for much smaller
history allocations when IDs have gaps. They do not measure response copying.

Threaded result reads now return a defensive deep copy, consistent with inline
tasks. Mutating a returned dictionary or nested list cannot alter subsequent
reads. Inline result copies run outside the task lock so custom copy hooks can
query status without deadlocking. Copy cost scales with result size; adapters
must return values that support deep copying.

Focused regression checks:

```bash
python -m pytest -q tests/test_request_history.py tests/test_task.py
```

## Pilot fault isolation and result reads

TaskPilot now uses safe exception formatting when collecting responses, accepting
requests, checking motion synchronization, and querying health. A broken SDK
exception formatter produces an error record instead of terminating dispatch or
preventing other services from appearing in the health snapshot.

`query_request()` and `wait_request()` copy captured request records outside the
scheduler condition lock. Slow result copies no longer block unrelated request
submission or completion. Waiters continue to protect records from pruning until
copying finishes and are released even if copying fails. Copy errors also retain
their original message instead of a payload `KeyError` being misreported as an
unknown request.

```bash
python -m pytest -q tests/test_task_pilot.py
```

## Callback waits and shared service lifecycles

Scheduler callbacks now receive `RuntimeError` if a request or health wait would
block their own pilot's scheduler. This covers grouped request waits as well.
Completed results, empty groups, and zero-timeout queries remain supported.
Ordinary callers on other threads keep their existing waiting behavior. The
guard detects direct scheduler reentry, not arbitrary dependency cycles between
application threads or device SDK calls.

Multiple service names may still refer to the same task instance. Initialization,
completion callback registration, rollback, and shutdown now operate once per
instance, in first-registration order. Completion notifications continue to wake
requests for every alias. Distinct instances remain independently managed even
if their classes or equality behavior match.

## Amortized result pruning

Explicit result removal now updates a live-ID set instead of rebuilding the
entire eviction heap per call. Stale heap entries are skipped during eviction;
the heap is rebuilt when its size exceeds twice the live count plus 64. Removing
the last live result clears the heap immediately. Only completed results enter
this bookkeeping, and eviction still retains the largest request IDs.

Local Python 3.10 measurements, median of three runs against the working copy
immediately before this change (not repository HEAD):

| Results pruned | Before seconds | After seconds |
| ---: | ---: | ---: |
| 1,000 | 0.018889 | 0.000782 |
| 5,000 | 0.451120 | 0.004126 |
| 10,000 | 1.862365 | 0.009000 |

Pruning timings exclude filling result storage. The existing 10,000-request
submission/completion benchmark with a 100-result limit changed from 0.041312 s
to 0.043063 s (about 4% additional time). The live-ID set adds bookkeeping memory;
its size is bounded by the configured result limit. These no-op timings exclude
device and network work.

The benchmark now includes a pruning scenario:

```bash
python -m examples.benchmark_task_storage --scenario pruning --counts 1000 5000 10000 --repeats 3
python -m pytest -q tests/test_result_retention.py tests/test_task.py
```

## Completed-outcome status accounting

Thread-pool tasks now account for retained success, failure, and cancellation
outcomes when completion is observed. Status reads only inspect futures whose
completion has not yet been recorded; completed history is not rescanned. A
status read can reconcile a completed future before its callback runs, without
double counting when the callback arrives. Eviction and explicit pruning remove
the corresponding counts. Public status fields retain their previous meaning:
counts describe retained results, not lifetime totals.

Local Python 3.10 median of three runs, comparing the working implementation
immediately before this change:

| Retained completed results | 100 status reads before | 100 status reads after |
| ---: | ---: | ---: |
| 1,000 | 0.063087 s | 0.000040 s |
| 10,000 | 0.750848 s | 0.000045 s |

These measurements exclude filling the storage. The 10,000-request no-op sparse
submission benchmark changed from 0.035473 s to 0.043032 s (about 21% additional
time); pruning changed from 0.010830 s to 0.012655 s. Accounting adds a map entry
per retained future and shifts outcome inspection to completion. Status reads
remain linear in the number of pending futures, but are constant-time when no
pending futures remain. This improves frequent monitoring with large completed
histories; it is not a claim of faster device execution.

```bash
python -m examples.benchmark_task_storage --scenario status --counts 1000 10000 --repeats 3
python -m pytest -q tests/test_task_status_accounting.py
```

## Submission and completion hot paths

A subsequent pass reduces the overhead added by accounting and retention:

- IDs greater than the latest accepted ID skip the duplicate-history lookup.
  Submission still commits history only after executor submission succeeds.
- A bitmap insertion only checks for watermark advancement when it fills the
  leading block.
- Outcome accounting uses one Future exception lookup; cancellation of the
  Future remains distinct from a callable raising `CancelledError`.
- At capacity, retention skips stale heap heads and uses `heappushpop` to replace
  the oldest live ID. Older late results can be rejected without live-set churn.

Local Python 3.10 median of seven runs, 10,000 requests, against the working copy
immediately before this pass (all three changed implementations snapshotted):

| Scenario | Before seconds | After seconds |
| --- | ---: | ---: |
| Sparse submission and immediate completion | 0.047948 | 0.040921 |
| Submission and completion with 100-result limit | 0.060550 | 0.047882 |
| Pruning only | 0.011899 | 0.010624 |
| 100 completed-history status queries | 0.000043 | 0.000043 |

The pruning implementation itself did not change in this pass; its timing
difference should be treated as measurement variation. Request history and
status semantics, snapshot isolation, and bookkeeping bounds are unchanged.

## Validate the payload that will be dispatched

`RequestSchema.validate()` now makes its defensive copy before validating field
constraints. It also verifies that the copy is a dictionary. Previously, a
custom copy hook could turn valid source data into an invalid returned command
after validation had already succeeded. Nested values had the same issue.

The schema now validates the exact returned snapshot. Regression tests cover
out-of-range values, unexpected fields, a non-object copy, and nested array
type changes. Invalid snapshots fail before a CallableTask invokes or retries
its device callable. This does not make copying arbitrary application objects
atomic with respect to concurrent writers; callers must still synchronize shared
mutable input data.

## Output-reference failure diagnostics

Runtime output-reference errors now identify the upstream node and the path
prefix that failed. Missing fields remain `KeyError`, invalid array indices
remain `IndexError`, and container/type mismatches remain `ValueError`.
Diagnostics report the binding location without embedding the upstream output.

Regression coverage now checks missing fields, short arrays, incompatible
containers, and traversal into scalar values. It verifies that the consumer is
not invoked, its resource claims are released without quarantine, and a later
workflow can use the same resource successfully.

## Release completed dispatch inputs before idle waits

The asynchronous scheduler previously retained its most recent dispatch locals
while awaiting another event. If another workflow remained active, these locals
could keep a completed run and its compiled input definition alive despite
normal run cleanup. This retained the last dispatch, rather than accumulating
every completed run.

A synchronous scheduler-turn method now contains those temporary references, so
they are released before the driver waits again. Weak-reference regression tests
cover successful and failed runs while another run remains active. Runs retained
by an explicit idempotency key remain available, while their completed compiled
inputs are still released.

## Isolate errors during bulk cancellation

`TaskPilot.cancel_all_requests()` now continues after an individual backend
cancellation raises an ordinary exception. Each failure is logged with its
request ID using safe exception formatting. The returned list still contains
only accepted cancellations, and a failed attempt is not falsely marked as
cancelled. Service filtering and unknown-service validation are unchanged.

Direct `cancel_request()` calls still propagate backend errors. A regression test
also confirms that completion observed during a failing cancellation is applied
after the cancellation attempt exits, without leaving cancellation bookkeeping
behind.

## Telemetry shutdown races

PollingTask status reads now capture the worker reference once. A concurrent
stop can clear the stored reference without causing an `AttributeError` halfway
through the status read.

A poll callback can also request stop while an external caller holds the
lifecycle lock and joins the worker. The worker-side stop sets the existing stop
event without waiting on that lock, avoiding a dependency cycle that previously
lasted until the external join timed out. External lifecycle operations remain
serialized. Deterministic tests pause a status read across shutdown and verify
that a worker-side stop completes while the external join is still active.

## Avoid repeated failure logs during result polling

Thread-pool failure logging now happens in the completion callback instead of
`peek_response()`. Repeated reads still return the error message but no longer
write the same log or format the exception twice per read. Logging and exception
formatting run outside the task lock. Completion notification is protected by
`finally`, so a failing log handler cannot suppress observer notification.

Tests verify one failure log across ten result reads, one formatting call per
read, formatter access to the task lock, and result/notification availability
when the log sink raises an exception.

## Isolate custom device status snapshots

CallableTask now deep-copies the custom status dictionary before merging it into
its status snapshot. Nested device dictionaries and lists are no longer shared
with the caller, so mutations on either side do not retroactively alter an
earlier snapshot. Copies run outside both task and retry locks. Custom status
objects must support deep copying; the copied result must remain a dictionary.

## Bound retry arithmetic for integer backoff values

RetryPolicy accepts integer backoff factors as well as floats. Exponentiation
now converts the factor to float so an extreme attempt index cannot construct
an enormous arbitrary-precision integer before the maximum delay is applied.
Overflow still follows the existing logarithmic fallback and delay cap.

A subprocess test with a three-second timeout verifies an integer factor of 2
at attempt `10**400`; isolating it prevents a regression from hanging the test
runner. Additional cases check parity between integer and floating-point factors.
