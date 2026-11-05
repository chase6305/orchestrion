# TaskPilot guide

`TaskPilot` coordinates one robot backend and a fixed map of peripheral services.
Start with the [README example](../README.md#move-then-grip) or run
`python -m examples.simulated_pick_and_place` from the repository root.

## Motion synchronization

Pass a `MoveSyncOption` to `pilot.call_srv_async(service, content, sync_option)`:

| Policy | Request becomes eligible when… |
| --- | --- |
| `no_sync()` | Submitted; no motion dependency |
| `sync_w_latest_move()` | The last move known at submission finishes |
| `sync_w_explicit_id(move_id)` | That specific main move finishes |
| `sync_w_submodule(name, move_id)` | That component's move finishes; omit the ID for its latest move |

`pilot.move_joint_trajectory_async()` returns a half-open `(begin, end)` range of
move IDs. For the final segment use `end - 1`; check `begin < 0` for rejection.
The latest-move policy captures the ID at request submission, so later motions do
not change its dependency. Higher integer request priorities run first among
eligible requests; ties preserve request-ID order. Running work is not preempted.

Built-in robots cancel unfinished main and submodule motions on `stop()`;
dependent service requests become `CANCELLED`. Completion and cancellation history
survive restart. Custom backends can report cancelled main moves through
`ReducedRobotState.cancelled_move_ids`.

Adapter request IDs and movement counters must be Python integers; boolean and float
aliases are rejected. The `-1` movement sentinel means no move yet. Invalid
submission counters reject new requests; invalid completion counters fail waiting
requests before a peripheral is invoked.

An unexpected built-in motion-worker failure stops local scheduling, cancels
unfinished move IDs and wakes waiters. Restart explicitly with `initialize()`
after the worker exits. Device adapters must handle physical stopping themselves.

For modular robots, `move_submodule_trajectory_async()` returns a completion-aware
move ID. `move_submodule_async()` remains a boolean compatibility wrapper.
`move_submodules_trajectories_async({name: path, ...})` validates every trajectory
before enqueuing any, and `wait_submodule_moves(moves, timeout=...)` shares one
deadline across them. See the [dual-arm examples](../examples/dual_arm/README.md).

## Results, cancellation and retention

Requests progress through `QUEUED`, optionally `WAITING_FOR_MOVE`, then `RUNNING`
and a terminal `SUCCEEDED`, `FAILED` or `CANCELLED` state.

```python
snapshot = pilot.query_request(request_id)
result = pilot.wait_request(request_id, timeout=1.0)
results = pilot.wait_requests(request_ids, timeout=2.0)
cancelled = pilot.cancel_request(request_id)
cancelled_ids = pilot.cancel_all_requests(service_name="camera")
events = pilot.timeline(request_id)
removed = pilot.prune_completed_requests(keep_last=1000)
```

Wait timeouts limit the caller's wait and do not cancel execution. Group waits
validate all IDs first, use one shared timeout and protect their results from
pruning while waiting. Cancellation is guaranteed before execution and best-effort
for running tasks; it does not prove a physical device has stopped.

An `idempotency_key` on `call_srv_async()` deduplicates retained requests for the
same service. Pruning a request releases its key. Local keys do not replace
protocol-level device deduplication after a lost network response. The
[network demo](../examples/network_service_demo.py) demonstrates both layers.

Task response caches and pilot history are independent. A positive
`max_result_count` on a function task bounds completed responses while preserving
pending work. It keeps the largest request IDs, regardless of completion order;
a negative limit is unlimited and zero is invalid. A response evicted before
collection becomes a failed request, so allow room for completion bursts.
Request-ID deduplication metadata remains for the task's lifetime, including
after response pruning.

Pruning a cancelled request also arranges response cleanup in built-in threaded
tasks once the SDK call finishes, even if the pilot has already stopped.

## Device adapters

| Adapter | Use |
| --- | --- |
| `CallableTask` | SDK calls and one-shot reads on the supplied executor |
| `PollingTask` | Background sampling; requests return the cached sample |
| `InPlaceFunctionCallTask` | Short local operations on the scheduler thread |
| `ThreadedPoolFunctionCallTask` | Custom threaded task subclasses |
| `GenericTask` | Custom request, result and lifecycle implementations |

Pass a `ThreadPoolExecutor` as `executor_owned` when using threaded tasks; the
pilot shuts it down on stop. Async here describes the request API: blocking I/O
still needs a worker. Completion notifications wake the scheduler; adapters
without notifications use `poll_interval` as a fallback.

`CallableTask(call, status=..., metadata=..., request_schema=...)` wraps a function
accepting `(request_id, content)` and returning a dictionary or `None`. A status
callback exposes device facts; metadata and schemas appear in
`pilot.describe_services()` without device I/O. Run
`python -m examples.simulated_multi_peripheral` for command, telemetry and health
adapters.

Threaded tasks report `health="offline"` until bound to an executor and after
stop. A `CallableTask` status callback may refine device health and availability;
`initialized` always describes the adapter's executor binding. Device facts cannot
make a stopped adapter available for new commands.

Use `RequestSchema` and `FieldSpec` to reject malformed commands before device I/O:

```python
from orchestrion import FieldSpec, RequestSchema

capture_input = RequestSchema({
    "exposure_ms": FieldSpec("number", required=True, minimum=0.1, maximum=100.0),
})
```

Arrays support `items`, `min_items` and `max_items`; nested objects use
`object_schema`. Errors include paths such as `objects[0].pose[1]`. Schemas
snapshot constraints at construction, preserve exact integer bounds and
distinguish JSON booleans from numbers in composite choices.

Retry is disabled by default. Set `retry_policy=RetryPolicy(...)` only for
idempotent reads or commands with protocol-level deduplication. Backoff is bounded
and interrupted by stop; an old invocation cannot resume retries after restart.
Device I/O needs its own bounded timeouts and truthful completion acknowledgement.

## Health and lifecycle

`query_task_status(name)` surfaces device errors; `query_all_task_statuses()` is
best-effort and represents a failing device as `{"available": False, "error": ...}`.
`query_health()` combines scheduler liveness, robot state, request counts and
peripheral statuses in a JSON-compatible snapshot.

The monotonic `revision` advances on pilot state transitions and adapter
notifications. `wait_health_change(revision, timeout=...)` waits for a newer
revision; timeout returns `None`. Passive device changes and sample aging do not
notify: `query_health()` reads current state, and `query_health(stale_after=...)`
marks stale online observations as degraded. Adapters use `DeviceHealth` values
(`unknown`, `connecting`, `online`, `degraded`, `offline`) and wall-clock
`observed_at` values.

Use `with TaskPilot(...) as pilot:` for lexical ownership, or pair `initialize()`
with `stop()` in `finally`. `is_running`, `service_names` and the defensive-copy
`task_map` expose lifecycle and registry state.

`stop(timeout=...)` bounds the scheduler-thread join. Owned executor shutdown
does not wait for already-running callbacks; device shutdown hooks need their own
timeouts and callbacks must exit cooperatively. After a join timeout, another
`stop()` can wait for the worker. Restart is rejected until old workers exit, and
an owned executor cannot be reused after shutdown. Built-in robots reject motion
while stopped; submissions prepared across a stop/restart boundary are rejected.

Configured sampling and scheduler intervals, trajectory point intervals and stop
timeouts must fit `threading.TIMEOUT_MAX`. Unsupported values are rejected before
queue or lifecycle changes. Deadline-based request and movement waits retain
larger total budgets by splitting native waits. `RetryPolicy` can compute larger
delays, but a callable task rejects an unsupported delay before waiting.
