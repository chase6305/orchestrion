# Experimental workflows

`orchestrion.workflow` coordinates device-independent DAGs in one event loop.
Use it for dependencies and shared resources across arms, fixtures, vision,
speech or mobile bases. See the [architecture and migration design
(中文)](architecture-v3.md) for scope and roadmap. Each physical device needs one
command owner when migrating from TaskPilot.

## Start with a simulated cell

```bash
python -m examples.heterogeneous_workflow --preview
python -m examples.heterogeneous_workflow
```

The planner builds a graph from a capability catalog; the engine validates it
before dispatch. `Action` registers a handler and its minimum resource claims,
`Resource` defines capacity, and `Node` calls an action. `after` adds ordering;
`OutputRef` adds a dependency and binds a producer's result into an input.

```python
import asyncio

from orchestrion.workflow import Action, Node, OutputRef, Workflow, WorkflowEngine


async def detect(context, inputs):
    return {"pose": [0.4, 0.1, 0.2]}


async def inspect(context, inputs):
    return {"inspected": inputs["pose"]}


async def main():
    async with WorkflowEngine([
        Action("detect", detect), Action("inspect", inspect),
    ]) as engine:
        run = engine.submit(Workflow((
            Node("detect", "detect"),
            Node("inspect", "inspect", {"pose": OutputRef("detect", ("pose",))}),
        )))
        print((await run.wait(timeout=2)).status.value)


asyncio.run(main())
```

Real devices declare resource claims; capacities apply across every run in an
engine. `BlockingAction` wraps blocking SDK calls in a supplied or default
`ThreadPoolExecutor`. Queued wrappers check cancellation before device I/O;
already-started calls drain while retaining claims. Async handlers must keep the
event loop free. Express ordering through graph dependencies: an action cannot
wait on a run or close its own engine.

Unexpected coroutine cancellation also keeps a running `BlockingAction` in
engine accounting until its SDK call exits. Its cancellation token prevents queued
wrappers from entering device I/O. Configure `stop_timeout` to report `UNKNOWN`
after a bounded stop wait; the still-draining SDK call keeps its concurrency slot
and resource claims, and `confirm_quiescent()` remains blocked until it exits.

## Contracts and bounds

Actions accept `input_schema` and `output_schema` using `RequestSchema` and
`FieldSpec`. Compilation checks known inputs even when other fields contain
references, and rejects structurally impossible output paths. Bound inputs get
full validation immediately before dispatch. Output violations fail the producer
with `ActionOutputError` and skip its dependants; retained JSON remains available
for diagnosis. Optional or dynamic data still requires runtime validation.

Optional engine limits `max_input_bytes_per_node`, `max_output_bytes_per_node`
and `max_feedback_bytes_per_report` measure compact ASCII JSON. Inputs are checked
before admission and again after schema validation of resolved bindings. Final
schema payloads are isolated finite JSON objects and remain subject to byte limits.
Oversized outputs fail before retention or downstream dispatch. Keep images and
point clouds in artifact storage and pass references; these limits do not bound
total process memory.

`context.report()` accepts JSON progress. Reports on the owning event loop are
delivered immediately; SDK-thread reports use a per-action bounded queue
(`max_pending_feedback_per_action=128` by default). Overflow drops the oldest
pending report and increments `NodeResult.feedback_dropped`. Late reports are
dropped. Required results belong in final outputs or the device protocol.

Optional progress callback failures are logged and counted in
`feedback_dropped`; they cannot rewrite an action result or release its device
claims. A failed loop notification drops and counts the pending batch, then the
next SDK report retries notification. Repeated notification failures log once
until notification recovers. Completion flushes accepted progress, including
the remainder of an in-flight delivery batch, before the final state.

## Cancellation and diagnostics

Handlers can return an `ActionOutcome` to report execution failures without
raising inside asynchronous work:

```python
from orchestrion.workflow import ActionOutcome


async def grip(context, inputs):
    if context.cancel_requested:
        return ActionOutcome.cancelled(context.cancel_reason)
    observation = await device.close_and_wait(inputs)
    if observation.stop_unconfirmed:
        return ActionOutcome.unknown("gripper stop could not be confirmed")
    if observation.failed:
        return ActionOutcome.failed("gripper rejected the command")
    return {"closed": True}
```

`FAILED` and `CANCELLED` outcomes confirm that device work has stopped. `UNKNOWN`
keeps claims quarantined and makes the run `NEEDS_ATTENTION`. The engine records
the outcome before logging failures; logging errors cannot interrupt accounting.
Returning a normal JSON value still means success. An explicit
`ActionOutcome(NodeStatus.SUCCEEDED, output=...)` follows the same JSON budget,
schema validation and defensive copying. Failure diagnostic outputs are bounded;
an invalid diagnostic never changes `UNKNOWN` or releases its leases.

Built-in blocking, ticket and composite adapters return these outcomes for known
failures and cancellations. Existing adapters that raise `ActionCancelled`,
`ExecutionUncertain` or ordinary SDK exceptions remain supported at the execution
boundary. The legacy `context.raise_if_cancelled()` and interruptible
`context.sleep()` retain their raising contract; return a cancellation outcome
after cleanup when writing new adapters. Synchronous definition/submission
validation and caller wait timeouts retain their existing exception contracts.

`run.wait(timeout=...)` only bounds the caller's wait. `Workflow(timeout=...)`
covers dependencies, resource queues and execution; `Node(timeout=...)` begins at
execution. Expiry stops admission and requests cooperative cancellation. With
`FAIL_FAST`, node expiry stops sibling admission before cleanup finishes.

Claims stay held until adapters confirm completion. An unknown outcome records
`UNKNOWN` and quarantines claims. Neither command acceptance nor cancellation
acceptance proves physical termination. After external reconciliation, call
`engine.confirm_quiescent(run.operation_id(node_id), evidence="...")`; the
historical result remains `UNKNOWN`.

`stop_timeout` on the engine bounds stop acknowledgement; a node override takes
precedence, and `None` inherits the engine default (unbounded). Expiry marks the
node `UNKNOWN` without cancelling its adapter or releasing resources. A draining
adapter keeps its concurrency slot and must exit before reconciliation.
`aclose(timeout=...)` bounds the combined wait for runs and draining adapters after
issuing stop requests. A timeout keeps active claims held; call it again to await
cleanup.

Use `run.explain(node_id)` for dependency, concurrency, resource, queue, stop or
quarantine blockers. `engine.scheduler_snapshot()` shows counts, owners, queue
heads and errors without device calls. A scheduler fault closes admission,
fails unstarted work and requests running work to stop; cleanup still retains
claims until confirmed quiescent.

## Submission identity and retention

`submit(workflow, idempotency_key="cycle-42")` returns the original retained run
for identical submissions, including failed or cancelled results. Changed
execution snapshots raise `ValueError`. Validation happens before reserving a
key; replay does not restart deadlines. Retrieve handles with
`engine.get_run_by_key(key)`.

`max_idempotency_keys` defaults to 1024. Explicitly use
`engine.prune_idempotency_keys(keep_last=...)` to forget settled keys; active and
unreconciled runs are preserved. Pruned keys can execute again. Unkeyed completed
runs are not retained unless they contain unknown operations.

Set `Action(..., version="driver-v2")` when capability semantics change. Versions
participate in submission identity; handler implementations are not hashed.

## Durable execution and device tickets

Run `python -m examples.durable_device_workflow` for a loopback HTTP example that
grips, releases, reopens its journal and replays without another device command.
Use `--journal /path/to/cell.sqlite3` to retain history and `--cycle-key ...` for a
new business cycle.

The application owns `SQLiteJournal`; keep it outside the engine context:

```python
with SQLiteJournal("cell.sqlite3") as journal:
    async with WorkflowEngine(actions, resources, journal=journal) as engine:
        run = engine.submit(workflow, idempotency_key="cycle-42")
        result = await run.wait(timeout=30)
```

Admission and dispatch intent commit before device I/O. Completed keyed results
replay across restarts. Interrupted dispatched operations become `UNKNOWN` and
keep their claims; unstarted nodes become `CANCELLED`. Recovery does not resume
the graph or resend actuator commands. Missing resources or insufficient capacity
reject startup. Storage failure closes admission and requests cooperative stop;
inspect `journal_error` in the scheduler snapshot.

Use a local filesystem and one journal/engine per device domain. The journal
holds an exclusive lifetime lock; retain its `.owner` sidecar while alive and do
not inherit connections across `fork`. Automatic journal mode uses WAL/FULL on
SQLite versions with the WAL-reset fix, otherwise rollback/EXTRA. `"delete"`
forces rollback; explicit `"wal"` requires the fix. Use SQLite backup APIs for
live backups. Synchronous commits run on the event loop; storage is not real-time.
Close and drain the engine before closing the journal.

`journal.get_run(run_id)` reads results, tickets, claims and reconciliation
evidence. `journal.events(run_id, after=0)` reads bounded audit history; progress
stays in memory. Engine key pruning commits to disk.
`journal.prune_completed_runs(keep_last=...)` removes settled unkeyed history,
preserving unresolved operations; it does not promise a smaller database file.

For ticket devices, implement async `OperationBackend.submit/query/cancel` and
bind `TicketAction(backend)` as the action handler. It sends
`context.operation_id` as the protocol key, persists accepted tickets with
`context.checkpoint()` and polls for final results. Submit is never retried
automatically. Lost submit responses, unavailable results and missing stop
confirmation become `UNKNOWN`. Backend I/O must be bounded. Checkpoints replace
previous metadata and have a 64 KiB compact ASCII JSON limit; a draining adapter
can persist a late ticket without changing the outcome or releasing its lease.

RPC helper tasks close rejected coroutines and drain accepted work before
returning an error outcome, including when cancellation interrupts cleanup again.
An unstarted submit task is a known failure with no device effects; a factory
failure after submit started remains `UNKNOWN`. Accepted tickets returned during
timeout cancellation cleanup are still checkpointed. A failed cancellation RPC
can be retried idempotently while final device observations remain authoritative.

## Composite capabilities

`workflow_action(name, inner_workflow, actions, resources)` wraps a fixed DAG as
one action. It reserves each listed resource's full capacity before any inner
step and keeps continuous ownership. Inner steps follow their own dependencies
and capacity constraints; an unknown step quarantines the whole scope.

Cancellation reaches nested scopes synchronously through loop-local callbacks;
it does not depend on starting a forwarding task. Deep nesting uses an iterative
dispatch queue. Inner handlers recheck parent cancellation before device entry,
and callbacks are released when the scope finishes.

The output contains `nodes[step_id]` with `status`, `output` and `error`. Use
`run.operation_id(node_id)` for encoded scoped IDs. Inner plans, action versions
and limits contribute to the composite version. Bind parameters when constructing
the inner graph; dynamic node inputs are not supported. Outer concurrency counts
composites and inner concurrency counts steps. Use shared resource claims to
enforce aggregate capacity across scopes.
