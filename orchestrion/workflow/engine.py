"""Event-driven, single-event-loop execution with engine-wide resource ownership."""

import asyncio
import copy
import heapq
import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple

from orchestrion.schema import CommandValidationError

from .actions import Action, ActionContext
from .compiler import (
    CompiledNode,
    check_claims,
    compile_workflow,
    describe_inputs,
    merge_claims,
    resolve_inputs,
    submission_fingerprint,
)
from .diagnostics import explain_node, scheduler_snapshot
from .model import (
    ActionCancelled,
    ActionOutputError,
    ExecutionUncertain,
    FailurePolicy,
    NodeResult,
    NodeStatus,
    Resource,
    ResourceClaim,
    RunResult,
    RunStatus,
    Workflow,
    WorkflowEvent,
    _name,
    _positive_int,
    _snapshot,
    _timeout,
)

_current_execution: ContextVar = ContextVar(
    "orchestrion_workflow_execution", default=None
)


def _error_text(error: BaseException) -> str:
    """A broken SDK exception formatter must not strand the scheduler."""
    try:
        message = str(error)
    except Exception:
        message = "exception message is unavailable"
    return "{}: {}".format(type(error).__name__, message)


class WorkflowRun:
    """A retained run handle; waiting never cancels the underlying operations."""

    def __init__(
        self,
        engine: "WorkflowEngine",
        workflow: Workflow,
        compiled: Tuple[CompiledNode, ...],
        idempotency_key: Optional[str] = None,
    ) -> None:
        self._engine = engine
        self._run_id = uuid.uuid4().hex
        self._idempotency_key = idempotency_key
        self._deadline = (
            None if workflow.timeout is None else engine._loop.time() + workflow.timeout
        )
        self._deadline_timer = None
        self._deadline_expired = False
        self._by_id = {item.node.node_id: item for item in compiled}
        self._remaining_dependencies = {
            item.node.node_id: len(item.dependencies) for item in compiled
        }
        self._successors = {node_id: [] for node_id in self._by_id}
        for item in compiled:
            for dependency in item.dependencies:
                self._successors[dependency].append(item.node.node_id)
        self._unfinished = len(compiled)
        self._policy = workflow.failure_policy
        self._results = {
            item.node.node_id: NodeResult(item.node.node_id, item.node.action)
            for item in compiled
        }
        self._contexts: Dict[str, ActionContext] = {}
        self._events = deque(maxlen=engine._max_events)
        self._sequence = 0
        self._cancelled = False
        self._failed_fast = False
        self._status = RunStatus.RUNNING
        self._done = engine._loop.create_future()
        for node_id in self._results:
            self._emit(node_id, "state")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def idempotency_key(self) -> Optional[str]:
        return self._idempotency_key

    @property
    def done(self) -> bool:
        return self._done.done()

    def snapshot(self) -> RunResult:
        """Return a defensive snapshot; terminal UNKNOWN is not a stop confirmation."""
        return RunResult(self.run_id, self._status, copy.deepcopy(self._results))

    def explain(self, node_id: str) -> Dict[str, Any]:
        """Describe a node's current wait or unresolved stop without device I/O."""
        self._engine._check_loop()
        _name(node_id, "node ID")
        return explain_node(self, node_id)

    def events(self, after: int = 0) -> Tuple[WorkflowEvent, ...]:
        """Read bounded history; a sequence gap indicates evicted events."""
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("after must be a nonnegative integer")
        return tuple(
            copy.deepcopy(event) for event in self._events if event.sequence > after
        )

    def cancel(self) -> bool:
        """Request cooperative stop. Running nodes keep their claims until quiescent."""
        self._engine._check_loop()
        if self.done or not self._unfinished or self._cancelled:
            return False
        self._cancelled = True
        self._status = RunStatus.CANCELLING
        self._engine._stop_run(self, "cancelled")
        return True

    async def wait(self, timeout: Optional[float] = None) -> RunResult:
        self._engine._check_loop()
        _timeout(timeout)
        if not self.done:
            self._engine._check_external_wait()
        waiter = asyncio.shield(self._done)
        if timeout is None:
            await waiter
        else:
            await asyncio.wait_for(waiter, timeout)
        return self.snapshot()

    def _emit(self, node_id: str, kind: str, detail: Any = None) -> None:
        self._sequence += 1
        self._events.append(
            WorkflowEvent(
                self._sequence,
                self.run_id,
                node_id,
                kind,
                self._results[node_id].status,
                time.time(),
                _snapshot(detail),
            )
        )

    def _transition(self, node_id: str, status: NodeStatus, **changes) -> None:
        if status.terminal:
            changes["finished_at"] = time.time()
        self._results[node_id] = replace(
            self._results[node_id], status=status, **changes
        )
        self._emit(node_id, "state", changes.get("error"))


class WorkflowEngine:
    """Coordinate heterogeneous actions without a privileged main robot.

    Use from one asyncio event loop. All plans share capacities within this engine.
    Handlers must be asynchronous and must confirm termination before completing;
    wrap blocking calls in BlockingAction. This is not a real-time controller or
    a distributed lock service. A closed engine cannot be restarted.
    """

    def __init__(
        self,
        actions: Iterable[Action],
        resources: Iterable[Resource] = (),
        *,
        max_concurrency: int = 32,
        max_active_runs: int = 128,
        max_nodes_per_run: int = 10000,
        max_events_per_run: int = 1000,
        max_idempotency_keys: int = 1024,
        max_pending_feedback_per_action: int = 128,
    ) -> None:
        for label, value in (
            ("max_concurrency", max_concurrency),
            ("max_active_runs", max_active_runs),
            ("max_nodes_per_run", max_nodes_per_run),
            ("max_events_per_run", max_events_per_run),
            ("max_idempotency_keys", max_idempotency_keys),
            ("max_pending_feedback_per_action", max_pending_feedback_per_action),
        ):
            _positive_int(value, label)
        self._resources: Dict[str, Resource] = {}
        for resource in resources:
            if not isinstance(resource, Resource):
                raise TypeError("resources must contain Resource objects")
            if resource.name in self._resources:
                raise ValueError("duplicate resource: " + resource.name)
            self._resources[resource.name] = resource
        self._actions: Dict[str, Action] = {}
        for action in actions:
            if not isinstance(action, Action):
                raise TypeError("actions must contain Action objects")
            if action.name in self._actions:
                raise ValueError("duplicate action: " + action.name)
            claims = merge_claims(action.claims)
            check_claims(claims, self._resources)
            self._actions[action.name] = replace(
                action,
                claims=claims,
                input_schema=copy.deepcopy(action.input_schema),
                output_schema=copy.deepcopy(action.output_schema),
            )
        self._max_concurrency = max_concurrency
        self._max_active_runs = max_active_runs
        self._max_nodes = max_nodes_per_run
        self._max_events = max_events_per_run
        self._max_idempotency_keys = max_idempotency_keys
        self._max_pending_feedback = max_pending_feedback_per_action
        self._submissions: Dict[str, Tuple[str, WorkflowRun]] = {}
        self._used = dict.fromkeys(self._resources, 0)
        self._leases: Dict[str, Tuple[ResourceClaim, ...]] = {}
        self._quarantined: Dict[str, Tuple[WorkflowRun, str]] = {}
        self._runs: Dict[str, WorkflowRun] = {}
        self._workers = set()
        self._running_count = 0
        self._ready_sequence = 0
        self._ready_heap = []
        self._ready_queued = set()
        self._resource_waiters = {name: [] for name in self._resources}
        self._loop = None
        self._wake = None
        self._idle = None
        self._driver = None
        self._starting_driver = False
        self._driver_error: Optional[str] = None
        self._closed = False

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            self._wake = asyncio.Event()
            self._idle = asyncio.Event()
            self._idle.set()
        elif self._loop is not loop:
            raise RuntimeError(
                "WorkflowEngine must be used from its original event loop"
            )

    def _check_external_wait(self) -> None:
        execution = _current_execution.get()
        if execution is not None and execution[0] is self and execution[1]._active:
            raise RuntimeError(
                "actions cannot wait on or close their own WorkflowEngine; "
                "express ordering with workflow dependencies"
            )

    def _spawn(self, coroutine):
        try:
            return self._loop.create_task(coroutine)
        except BaseException:
            coroutine.close()
            raise

    def _driver_done(self, driver: asyncio.Future) -> None:
        exception = None if driver.cancelled() else driver.exception()
        if driver is not self._driver or self._driver_error is not None:
            return
        if not driver.cancelled() and exception is None and not self._runs:
            return
        if driver.cancelled():
            self._driver_error = "CancelledError: workflow scheduler interrupted"
        elif exception is not None:
            self._driver_error = _error_text(exception)
        else:
            self._driver_error = "RuntimeError: workflow scheduler exited with active runs"
        self._closed = True
        for run in tuple(self._runs.values()):
            self._stop_run(run, "scheduler_failed")
            for node_id, context in tuple(run._contexts.items()):
                if (
                    not context._handler_started
                    and not run._results[node_id].status.terminal
                ):
                    # Failure between reservation and task startup may leave no
                    # worker to acknowledge cancellation. No adapter was entered;
                    # any already queued worker sees the cancellation before I/O.
                    self._settle(
                        run, run._by_id[node_id], context, NodeStatus.FAILED,
                        error="scheduler_failed",
                    )
            if not run.done and run._unfinished == 0:
                self._finish_run(run)

    def _check_driver(self) -> None:
        # Eager factories can enter adapters before the new driver is assigned.
        # During that interval the reference may still denote the previous run.
        if (
            not self._starting_driver
            and self._driver is not None
            and self._driver.done()
        ):
            self._driver_done(self._driver)

    def describe(self) -> Dict[str, Any]:
        """Return a side-effect-free capability catalog for planners and tooling."""
        return {
            "resources": {
                name: {"capacity": item.capacity}
                for name, item in self._resources.items()
            },
            "actions": {
                name: {
                    "claims": {claim.resource: claim.units for claim in action.claims},
                    "input_schema": action.input_schema.describe()
                    if action.input_schema
                    else None,
                    "output_schema": action.output_schema.describe()
                    if action.output_schema
                    else None,
                }
                for name, action in self._actions.items()
            },
        }

    def validate(self, workflow: Workflow) -> Dict[str, Any]:
        """Compile a proposal without starting actions; return its normalized graph."""
        compiled = compile_workflow(
            workflow, self._actions, self._resources, self._max_nodes
        )
        return {
            "failure_policy": workflow.failure_policy.value,
            "timeout": workflow.timeout,
            "nodes": [
                {
                    "node_id": item.node.node_id,
                    "action": item.node.action,
                    "inputs": describe_inputs(item.node.inputs),
                    "after": list(item.dependencies),
                    "claims": {claim.resource: claim.units for claim in item.claims},
                    "priority": item.node.priority,
                    "timeout": item.node.timeout,
                }
                for item in compiled
            ],
        }

    def submit(
        self, workflow: Workflow, *, idempotency_key: Optional[str] = None
    ) -> WorkflowRun:
        """Validate and admit a graph, or return its retained identical submission.

        Keys belong to this engine and are never evicted automatically. Reusing a
        key with a changed execution snapshot raises ValueError before device I/O.
        """
        self._check_loop()
        # A completed task's callback may still be queued. Observe failure before
        # admitting another command or replacing the driver reference.
        self._check_driver()
        if self._closed:
            raise RuntimeError("WorkflowEngine is closed")
        if idempotency_key is not None:
            _name(idempotency_key, "idempotency_key")
        compiled = None
        fingerprint = None
        if idempotency_key is not None:
            compiled = compile_workflow(
                workflow, self._actions, self._resources, self._max_nodes
            )
            fingerprint = submission_fingerprint(workflow, compiled)
            retained = self._submissions.get(idempotency_key)
            if retained is not None:
                if retained[0] != fingerprint:
                    raise ValueError(
                        "idempotency_key already belongs to a different workflow"
                    )
                return retained[1]
            if len(self._submissions) >= self._max_idempotency_keys:
                raise RuntimeError(
                    "max_idempotency_keys reached; explicitly prune completed keys"
                )
        if len(self._runs) >= self._max_active_runs:
            raise RuntimeError(
                "max_active_runs reached; wait for an active run to finish"
            )
        if compiled is None:
            compiled = compile_workflow(
                workflow, self._actions, self._resources, self._max_nodes
            )
        run = WorkflowRun(self, workflow, compiled, idempotency_key)
        self._runs[run.run_id] = run
        self._idle.clear()
        if idempotency_key is not None:
            self._submissions[idempotency_key] = (fingerprint, run)
        for item in compiled:
            if not item.dependencies:
                self._make_ready(run, item)
        try:
            if run._deadline is not None:
                run._deadline_timer = self._loop.call_at(
                    run._deadline, self._expire_run, run
                )
            if self._driver is None or self._driver.done():
                self._starting_driver = True
                try:
                    self._driver = self._spawn(self._schedule())
                finally:
                    self._starting_driver = False
                self._driver.add_done_callback(self._driver_done)
        except BaseException:
            self._stop_run(run, "scheduler_start_failed")
            self._finish_run(run)
            if idempotency_key is not None:
                self._submissions.pop(idempotency_key)
            raise
        self._wake.set()
        return run

    def get_run_by_key(self, idempotency_key: str) -> Optional[WorkflowRun]:
        """Look up a retained run, including after the engine has closed."""
        self._check_loop()
        _name(idempotency_key, "idempotency_key")
        retained = self._submissions.get(idempotency_key)
        return None if retained is None else retained[1]

    def prune_idempotency_keys(self, keep_last: int = 1000) -> int:
        """Forget old settled submissions, explicitly allowing those keys to run again.

        Retain the newest keep_last eligible keys by submission order. Active runs
        and runs with unreconciled UNKNOWN operations cannot be pruned. Existing
        run handles remain valid. This only affects engine-owned in-memory history.
        """
        self._check_loop()
        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 0:
            raise ValueError("keep_last must be a nonnegative integer")
        uncertain = {run.run_id for run, _ in self._quarantined.values()}
        eligible = [
            key
            for key, (_, run) in self._submissions.items()
            if run.done and run.run_id not in uncertain
        ]
        removed = eligible[: max(0, len(eligible) - keep_last)]
        for key in removed:
            del self._submissions[key]
        return len(removed)

    def scheduler_snapshot(self) -> Dict[str, Any]:
        """Read JSON-compatible counts, resource owners/queues and key retention.

        This scans current records on demand and does not poll any device. Counts
        describe active runs; completed UNKNOWN leases remain visible separately.
        """
        self._check_loop()
        return scheduler_snapshot(self)

    def resource_snapshot(self) -> Dict[str, Any]:
        """Expose occupied capacity and unresolved operations, including completed runs."""
        return {
            "resources": {
                name: {"capacity": resource.capacity, "used": self._used[name]}
                for name, resource in self._resources.items()
            },
            "quarantined": {
                operation: {
                    "run_id": run.run_id,
                    "node_id": node_id,
                    "claims": {
                        claim.resource: claim.units for claim in self._leases[operation]
                    },
                }
                for operation, (run, node_id) in self._quarantined.items()
            },
        }

    def confirm_quiescent(self, operation_id: str, *, evidence: str) -> None:
        """Release an UNKNOWN operation after external verification that it stopped.

        The caller is responsible for trustworthy device/operator evidence. This
        never resends a command or changes its historical UNKNOWN outcome.
        """
        self._check_loop()
        _name(evidence, "quiescence evidence")
        if operation_id not in self._quarantined:
            raise KeyError(operation_id)
        run, node_id = self._quarantined.pop(operation_id)
        self._release(operation_id)
        run._emit(node_id, "reconciled", {"evidence": evidence})
        self._wake.set()

    async def aclose(self, timeout: Optional[float] = None) -> None:
        """Stop admission and drain actions; timeout does not release active claims.

        May be called again after timeout. Uncertain operations remain quarantined
        and must be reconciled externally even after this method returns.
        """
        self._check_loop()
        _timeout(timeout)
        self._check_external_wait()
        self._check_driver()
        self._closed = True
        for run in tuple(self._runs.values()):
            run.cancel()
        # Only this notification wait is cancelled on timeout or caller exit.
        # Workers can still drain after an interrupted scheduler has terminated.
        if timeout is None:
            await self._idle.wait()
        else:
            await asyncio.wait_for(self._idle.wait(), timeout)

    async def __aenter__(self) -> "WorkflowEngine":
        self._check_loop()
        if self._closed:
            raise RuntimeError("WorkflowEngine is closed")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()

    def _stop_node(self, run: WorkflowRun, node_id: str, reason: str) -> None:
        result = run._results[node_id]
        if result.status.terminal:
            return
        context = run._contexts.get(node_id)
        if context is None:
            status = {
                "upstream_failure": NodeStatus.SKIPPED,
                "workflow_timeout": NodeStatus.TIMED_OUT,
                "scheduler_failed": NodeStatus.FAILED,
            }.get(reason, NodeStatus.CANCELLED)
            # A whole-run stop assigns its own cause to every unstarted node;
            # graph traversal must not make outcomes depend on declaration order.
            self._complete_node(run, node_id, status, propagate=False, error=reason)
        elif context._request_cancel(reason):
            run._transition(node_id, NodeStatus.CANCELLING, error=reason)
        self._wake.set()

    def _stop_run(self, run: WorkflowRun, reason: str) -> None:
        if run._deadline_timer is not None:
            run._deadline_timer.cancel()
        run._status = RunStatus.CANCELLING
        for node_id in run._results:
            self._stop_node(run, node_id, reason)
        # Cancelled entries behind a blocked device must not retain old runs
        # forever. Batch compaction is off the normal completion path.
        self._ready_heap = [
            entry for entry in self._ready_heap if self._is_ready(entry)
        ]
        heapq.heapify(self._ready_heap)
        self._ready_queued = {entry[1] for entry in self._ready_heap}
        for name, waiters in self._resource_waiters.items():
            self._resource_waiters[name] = [
                entry for entry in waiters if self._is_ready(entry)
            ]
            heapq.heapify(self._resource_waiters[name])
        self._wake_waiters(self._resources)
        self._wake.set()

    def _expire_run(self, run: WorkflowRun) -> None:
        if (
            run.done
            or not run._unfinished
            or run._deadline_expired
            or run._cancelled
            or run._failed_fast
        ):
            return
        run._deadline_expired = True
        self._stop_run(run, "workflow_timeout")

    def _expire_node(self, run: WorkflowRun, node_id: str) -> None:
        if run._results[node_id].status.terminal:
            return
        context = run._contexts[node_id]
        if context.cancel_requested:
            return
        self._stop_node(run, node_id, "timeout")
        if run._policy == FailurePolicy.FAIL_FAST and not run._failed_fast:
            run._failed_fast = True
            self._stop_run(run, "upstream_failure")

    def _check_run_deadline(self, run: WorkflowRun) -> None:
        if run._deadline is not None and self._loop.time() >= run._deadline:
            self._expire_run(run)

    def _release(self, operation_id: str) -> None:
        claims = self._leases.pop(operation_id)
        for claim in claims:
            self._used[claim.resource] -= claim.units
        self._wake_waiters(claim.resource for claim in claims)

    @staticmethod
    def _is_ready(entry) -> bool:
        return entry[2]._results[entry[3].node.node_id].status == NodeStatus.READY

    def _head(self, resource: str):
        waiters = self._resource_waiters[resource]
        while waiters and not self._is_ready(waiters[0]):
            heapq.heappop(waiters)
        return waiters[0] if waiters else None

    def _can_start(self, entry) -> bool:
        return all(
            self._head(claim.resource) is entry
            and self._used[claim.resource] + claim.units
            <= self._resources[claim.resource].capacity
            for claim in entry[3].claims
        )

    def _queue_candidate(self, entry) -> None:
        if entry[1] not in self._ready_queued and self._can_start(entry):
            heapq.heappush(self._ready_heap, entry)
            self._ready_queued.add(entry[1])

    def _wake_waiters(self, resources: Iterable[str]) -> None:
        if self._driver_error is not None:
            return
        for name in resources:
            entry = self._head(name)
            if entry is not None:
                self._queue_candidate(entry)

    def _make_ready(self, run: WorkflowRun, item: CompiledNode) -> None:
        run._transition(item.node.node_id, NodeStatus.READY)
        self._ready_sequence += 1
        entry = (
            -item.node.priority,
            self._ready_sequence,
            run,
            item,
        )
        for claim in item.claims:
            heapq.heappush(self._resource_waiters[claim.resource], entry)
        self._queue_candidate(entry)

    def _complete_node(
        self,
        run: WorkflowRun,
        node_id: str,
        status: NodeStatus,
        *,
        propagate: bool = True,
        **changes,
    ) -> None:
        run._transition(node_id, status, **changes)
        run._unfinished -= 1
        if not propagate:
            return
        completed = deque([node_id])
        # Visit affected graph edges once, rather than rescanning an entire DAG
        # after each completion. Iteration also supports very deep workflows.
        while completed:
            predecessor = completed.popleft()
            succeeded = run._results[predecessor].status == NodeStatus.SUCCEEDED
            for successor in run._successors[predecessor]:
                if run._results[successor].status != NodeStatus.PENDING:
                    continue
                if succeeded:
                    run._remaining_dependencies[successor] -= 1
                    if run._remaining_dependencies[successor] == 0:
                        self._make_ready(run, run._by_id[successor])
                else:
                    run._transition(
                        successor,
                        NodeStatus.SKIPPED,
                        error="dependency did not succeed",
                    )
                    run._unfinished -= 1
                    completed.append(successor)

    def _finish_run(self, run: WorkflowRun) -> None:
        if run._deadline_timer is not None:
            run._deadline_timer.cancel()
        statuses = {result.status for result in run._results.values()}
        if NodeStatus.UNKNOWN in statuses:
            run._status = RunStatus.NEEDS_ATTENTION
        elif statuses.intersection({NodeStatus.FAILED, NodeStatus.TIMED_OUT}):
            run._status = RunStatus.FAILED
        elif run._cancelled or NodeStatus.CANCELLED in statuses:
            run._status = RunStatus.CANCELLED
        else:
            run._status = RunStatus.SUCCEEDED
        self._runs.pop(run.run_id)
        # Definitions may contain large input data; handles only retain results
        # and bounded events after completion. Only explicit submission keys and
        # unresolved operations retain completed runs inside the engine.
        run._by_id.clear()
        run._remaining_dependencies.clear()
        run._successors.clear()
        run._contexts.clear()
        run._done.set_result(None)
        if not self._runs:
            self._idle.set()

    async def _schedule(self) -> None:
        while self._runs:
            self._wake.clear()
            self._advance_scheduler()
            if self._runs:
                await self._wake.wait()
        self._ready_heap.clear()
        self._ready_queued.clear()
        for waiters in self._resource_waiters.values():
            waiters.clear()

    def _advance_scheduler(self) -> None:
        """Advance one turn without retaining dispatch locals across idle waits."""
        for run in tuple(self._runs.values()):
            self._check_run_deadline(run)
            if run._unfinished == 0:
                self._finish_run(run)

        # Each resource exposes only its first eligible waiter. A single busy
        # arm with thousands of queued actions costs O(log N) per completion,
        # without repeatedly scanning the whole blocked backlog.
        while self._ready_heap and self._running_count < self._max_concurrency:
            entry = heapq.heappop(self._ready_heap)
            self._ready_queued.discard(entry[1])
            _, _, run, item = entry
            if not self._is_ready(entry) or not self._can_start(entry):
                continue
            node_id = item.node.node_id
            context = ActionContext(
                run.run_id,
                node_id,
                lambda data, r=run, n=node_id: r._emit(n, "feedback", data),
                max_pending_feedback=self._max_pending_feedback,
                feedback_overflow=lambda count, r=run, n=node_id: r._emit(
                    n, "feedback_overflow", {"dropped": count}
                ),
            )
            run._contexts[node_id] = context
            self._leases[context.operation_id] = item.claims
            for claim in item.claims:
                self._used[claim.resource] += claim.units
            self._running_count += 1
            run._transition(node_id, NodeStatus.RUNNING, started_at=time.time())
            self._wake_waiters(claim.resource for claim in item.claims)
            try:
                worker = self._spawn(self._execute(run, item, context))
            except Exception as exc:
                self._settle(
                    run,
                    item,
                    context,
                    NodeStatus.FAILED,
                    error="could not start action: " + _error_text(exc),
                )
            else:
                self._workers.add(worker)
                worker.add_done_callback(
                    partial(self._worker_done, run, item, context)
                )

    def _worker_done(
        self,
        run: WorkflowRun,
        item: CompiledNode,
        context: ActionContext,
        worker: asyncio.Future,
    ) -> None:
        self._workers.discard(worker)
        exception = None if worker.cancelled() else worker.exception()
        if run._results[item.node.node_id].status.terminal:
            return
        # A task cancelled before its coroutine starts cannot run its finally
        # block. No device was invoked in that case, so the reservation is free.
        if context._handler_started:
            context._request_cancel("adapter_interrupted")
            status = NodeStatus.UNKNOWN
        else:
            status = NodeStatus.CANCELLED if worker.cancelled() else NodeStatus.FAILED
        error = (
            _error_text(exception)
            if exception is not None
            else "action worker interrupted"
        )
        self._settle(run, item, context, status, error=error)

    async def _execute(
        self,
        run: WorkflowRun,
        item: CompiledNode,
        context: ActionContext,
    ) -> None:
        node = item.node
        timer = None
        node_deadline = None
        output = None
        error = None
        status = NodeStatus.SUCCEEDED
        execution_token = _current_execution.set((self, context))
        try:
            self._check_driver()
            if node.timeout is not None:
                node_deadline = self._loop.time() + node.timeout
                timer = self._loop.call_at(
                    node_deadline, self._expire_node, run, node.node_id
                )
            self._check_run_deadline(run)
            context.raise_if_cancelled()
            inputs = resolve_inputs(
                node.inputs,
                {key: run._results[key].output for key in item.dependencies},
            )
            action = self._actions[node.action]
            if action.input_schema:
                inputs = action.input_schema.validate(inputs)
            # Recheck after input preparation: a delayed event-loop timer must
            # not allow a new device command to slip past an expired budget.
            self._check_run_deadline(run)
            if node_deadline is not None and self._loop.time() >= node_deadline:
                self._expire_node(run, node.node_id)
            context.raise_if_cancelled()
            context._handler_started = True
            output = _snapshot(await action.handler(context, inputs))
            if action.output_schema is not None:
                # RequestSchema allows None as an empty command for legacy APIs;
                # a declared object result requires an actual JSON object.
                if not isinstance(output, dict):
                    raise ActionOutputError("action output must be a dictionary")
                try:
                    output = action.output_schema.validate(output)
                except CommandValidationError as exc:
                    raise ActionOutputError(str(exc)) from exc
        except ActionCancelled as exc:
            status = NodeStatus.CANCELLED
            error = _error_text(exc)
        except asyncio.CancelledError as exc:
            context._request_cancel("adapter_interrupted")
            status = NodeStatus.UNKNOWN
            error = _error_text(exc)
        except ExecutionUncertain as exc:
            status = NodeStatus.UNKNOWN
            error = _error_text(exc)
        except Exception as exc:
            status = NodeStatus.FAILED
            error = _error_text(exc)
        finally:
            if timer is not None:
                timer.cancel()
            _current_execution.reset(execution_token)

        # An adapter that monopolizes the loop can postpone timers; completion
        # still has to respect the elapsed budget. This cannot provide hard stop.
        self._check_run_deadline(run)
        if node_deadline is not None and self._loop.time() >= node_deadline:
            self._expire_node(run, node.node_id)
        self._settle(run, item, context, status, output=output, error=error)

    def _settle(
        self,
        run: WorkflowRun,
        item: CompiledNode,
        context: ActionContext,
        status: NodeStatus,
        *,
        output: Any = None,
        error: Optional[str] = None,
    ) -> None:
        node = item.node
        if run._results[node.node_id].status.terminal:
            return
        feedback_dropped = context._finish()
        if status != NodeStatus.UNKNOWN and context.cancel_requested:
            if context.cancel_reason in {"timeout", "workflow_timeout"}:
                status = NodeStatus.TIMED_OUT
            elif context.cancel_reason == "scheduler_failed":
                status = NodeStatus.FAILED
            else:
                status = NodeStatus.CANCELLED
            error = error or context.cancel_reason
        if status == NodeStatus.UNKNOWN:
            self._quarantined[context.operation_id] = (run, node.node_id)
        else:
            self._release(context.operation_id)
        self._running_count -= 1
        self._complete_node(
            run, node.node_id, status,
            output=output, error=error, feedback_dropped=feedback_dropped,
        )
        if (
            status
            in {
                NodeStatus.FAILED,
                NodeStatus.TIMED_OUT,
                NodeStatus.UNKNOWN,
                NodeStatus.CANCELLED,
            }
            and run._policy == FailurePolicy.FAIL_FAST
            and not run._failed_fast
            and not run._cancelled
        ):
            run._failed_fast = True
            self._stop_run(run, "upstream_failure")
        if self._driver_error is not None and run._unfinished == 0:
            self._finish_run(run)
        self._wake.set()
