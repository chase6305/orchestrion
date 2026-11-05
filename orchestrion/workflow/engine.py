"""Event-driven, single-event-loop execution with engine-wide resource ownership."""

import asyncio
import copy
import heapq
import time
import weakref
from collections import deque
from contextvars import ContextVar
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from .diagnostics import scheduler_snapshot
from .dispatch import DispatchBarrier
from .journal import JournalError, SQLiteJournal
from .model import (
    ActionCancelled,
    ActionOutputError,
    ExecutionUncertain,
    FailurePolicy,
    NodeStatus,
    Resource,
    ResourceClaim,
    RunStatus,
    Workflow,
    _name,
    _positive_int,
    _snapshot,
    _timeout,
)
from .recovery import restore_journal
from .run import WorkflowRun

# (-priority, sequence, run, node): descending priority, FIFO for ties.
_ReadyEntry = Tuple[int, int, WorkflowRun, CompiledNode]

_current_execution: ContextVar = ContextVar(
    "orchestrion_workflow_execution", default=()
)


def _error_text(error: BaseException) -> str:
    """A broken SDK exception formatter must not strand the scheduler."""
    try:
        message = str(error)
    except Exception:
        message = "exception message is unavailable"
    return "{}: {}".format(type(error).__name__, message)


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
        journal: Optional[SQLiteJournal] = None,
        stop_timeout: Optional[float] = None,
        max_input_bytes_per_node: Optional[int] = None,
        max_output_bytes_per_node: Optional[int] = None,
        max_feedback_bytes_per_report: Optional[int] = None,
    ) -> None:
        _timeout(stop_timeout)
        self._stop_timeout = stop_timeout
        self._stop_timers = {}
        self._draining = set()
        for label, limit in (
            ("max_input_bytes_per_node", max_input_bytes_per_node),
            ("max_output_bytes_per_node", max_output_bytes_per_node),
            ("max_feedback_bytes_per_report", max_feedback_bytes_per_report),
        ):
            if limit is not None:
                _positive_int(limit, label)
        self._max_input_bytes = max_input_bytes_per_node
        self._max_output_bytes = max_output_bytes_per_node
        self._max_feedback_bytes = max_feedback_bytes_per_report
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
        self._ready_heap: List[_ReadyEntry] = []
        self._ready_queued = set()
        self._resource_waiters: Dict[str, List[_ReadyEntry]] = {
            name: [] for name in self._resources
        }
        self._loop = None
        self._wake = None
        self._idle = None
        self._workers_drained = None
        self._driver = None
        self._starting_driver = False
        self._driver_error: Optional[str] = None
        self._closed = False
        if journal is not None and not isinstance(journal, SQLiteJournal):
            raise TypeError("journal must be SQLiteJournal or None")
        self._journal = journal
        self._dispatch_barrier = None if journal is None else DispatchBarrier(self)
        self._journal_error: Optional[str] = None
        self._journal_loaded = False
        if journal is not None:
            journal._bind(self)

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            self._wake = asyncio.Event()
            self._idle = asyncio.Event()
            self._idle.set()
            self._workers_drained = asyncio.Event()
            self._workers_drained.set()
            try:
                self._restore_journal()
            except Exception as exc:
                self._closed = True
                self._journal_error = _error_text(exc)
                raise
        elif self._loop is not loop:
            raise RuntimeError(
                "WorkflowEngine must be used from its original event loop"
            )

    def _restore_journal(self) -> None:
        if self._journal is None or self._journal_loaded:
            return
        restore_journal(self)
        self._journal_loaded = True

    def _record_journal(self, method: str, *args, strict: bool = False) -> None:
        if self._journal is None:
            return
        if self._journal_error is not None:
            if strict:
                raise JournalError(self._journal_error)
            return
        try:
            getattr(self._journal, method)(*args)
        except Exception as exc:
            self._journal_error = _error_text(exc)
            self._closed = True
            # Defer stop transitions until the current accounting update ends.
            self._loop.call_soon(self._stop_after_journal_failure)
            if strict:
                raise JournalError(self._journal_error) from exc

    def _stop_after_journal_failure(self) -> None:
        for run in tuple(self._runs.values()):
            self._stop_run(run, "journal_failed")
        self._wake.set()

    def _checkpoint_node(self, run: WorkflowRun, node_id: str, data: Any) -> None:
        self._record_journal("_checkpoint", run.run_id, node_id, data, strict=True)
        run._results[node_id] = replace(run._results[node_id], checkpoint=data)
        run._emit(node_id, "checkpoint", data)

    def _check_external_wait(self) -> None:
        for engine_ref, context_ref in _current_execution.get():
            context = context_ref()
            if (
                engine_ref() is self
                and context is not None
                and (context._active or context.operation_id in self._draining)
            ):
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
            self._driver_error = (
                "RuntimeError: workflow scheduler exited with active runs"
            )
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
                        run,
                        run._by_id[node_id],
                        context,
                        NodeStatus.FAILED,
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
                    "version": action.version,
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
            workflow,
            self._actions,
            self._resources,
            self._max_nodes,
            self._max_input_bytes,
        )
        return {
            "failure_policy": workflow.failure_policy.value,
            "timeout": workflow.timeout,
            "nodes": [
                {
                    "node_id": item.node.node_id,
                    "action": item.node.action,
                    "action_version": item.action_version,
                    "inputs": describe_inputs(item.node.inputs),
                    "after": list(item.dependencies),
                    "claims": {claim.resource: claim.units for claim in item.claims},
                    "priority": item.node.priority,
                    "timeout": item.node.timeout,
                    "stop_timeout": item.node.stop_timeout,
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
                workflow,
                self._actions,
                self._resources,
                self._max_nodes,
                self._max_input_bytes,
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
                workflow,
                self._actions,
                self._resources,
                self._max_nodes,
                self._max_input_bytes,
            )
        run = WorkflowRun(self, workflow, compiled, idempotency_key)
        self._record_journal("_admit", run, compiled, fingerprint, strict=True)
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
                self._record_journal("_prune_keys", (idempotency_key,), strict=True)
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
        run handles remain valid. With a journal, key removal is committed before
        it is forgotten in memory; unkeyed audit records can be pruned separately.
        """
        self._check_loop()
        if (
            isinstance(keep_last, bool)
            or not isinstance(keep_last, int)
            or keep_last < 0
        ):
            raise ValueError("keep_last must be a nonnegative integer")
        uncertain = {run.run_id for run, _ in self._quarantined.values()}
        eligible = [
            key
            for key, (_, run) in self._submissions.items()
            if run.done and run.run_id not in uncertain
        ]
        removed = eligible[: max(0, len(eligible) - keep_last)]
        if removed:
            self._record_journal("_prune_keys", removed, strict=True)
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
        recovery_pending = self._journal is not None and not self._journal_loaded
        return {
            "recovery_pending": recovery_pending,
            "resources": {
                name: {
                    "capacity": resource.capacity,
                    "used": None if recovery_pending else self._used[name],
                }
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
        if operation_id in self._draining:
            raise RuntimeError(
                "adapter is still draining; cannot release its resources"
            )
        run, node_id = self._quarantined[operation_id]
        self._record_journal("_reconcile", run.run_id, node_id, evidence, strict=True)
        self._quarantined.pop(operation_id)
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
        async def drain():
            await self._idle.wait()
            await self._workers_drained.wait()

        if timeout is None:
            await drain()
        else:
            await asyncio.wait_for(drain(), timeout)

    async def __aenter__(self) -> "WorkflowEngine":
        self._check_loop()
        if self._closed:
            raise RuntimeError("WorkflowEngine is closed")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()

    def _stop_node(
        self, run: WorkflowRun, node_id: str, reason: str, *, settlements=None
    ) -> None:
        result = run._results[node_id]
        if result.status.terminal:
            return
        context = run._contexts.get(node_id)
        if context is None:
            status = {
                "upstream_failure": NodeStatus.SKIPPED,
                "workflow_timeout": NodeStatus.TIMED_OUT,
                "scheduler_failed": NodeStatus.FAILED,
                "journal_failed": NodeStatus.FAILED,
            }.get(reason, NodeStatus.CANCELLED)
            # A whole-run stop assigns its own cause to every unstarted node;
            # graph traversal must not make outcomes depend on declaration order.
            self._complete_node(
                run,
                node_id,
                status,
                propagate=False,
                error=reason,
                settlements=settlements,
            )
        elif context._request_cancel(reason):
            run._transition(node_id, NodeStatus.CANCELLING, error=reason)
            budget = run._by_id[node_id].node.stop_timeout
            if budget is None:
                budget = self._stop_timeout
            if budget is not None:
                context._stop_deadline = self._loop.time() + budget
                self._stop_timers[context.operation_id] = self._loop.call_at(
                    context._stop_deadline,
                    self._stop_confirmation_expired,
                    run,
                    node_id,
                    context,
                )
        self._wake.set()

    def _stop_confirmation_expired(
        self, run: WorkflowRun, node_id: str, context: ActionContext
    ) -> None:
        if run._results[node_id].status.terminal:
            return
        if not context._handler_started:
            self._settle(run, run._by_id[node_id], context, NodeStatus.CANCELLED)
            return
        self._draining.add(context.operation_id)
        self._settle(
            run,
            run._by_id[node_id],
            context,
            NodeStatus.UNKNOWN,
            error="stop confirmation deadline expired; adapter may still be active",
            draining=True,
        )

    def _stop_run(self, run: WorkflowRun, reason: str) -> None:
        if run._deadline_timer is not None:
            run._deadline_timer.cancel()
        run._status = RunStatus.CANCELLING
        settlements = [] if self._journal is not None else None
        for node_id in run._results:
            self._stop_node(run, node_id, reason, settlements=settlements)
        if settlements:
            self._record_settlements(run, settlements)
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
    def _is_ready(entry: _ReadyEntry) -> bool:
        _, _, run, item = entry
        return run._results[item.node.node_id].status == NodeStatus.READY

    def _head(self, resource: str) -> Optional[_ReadyEntry]:
        waiters = self._resource_waiters[resource]
        while waiters and not self._is_ready(waiters[0]):
            heapq.heappop(waiters)
        return waiters[0] if waiters else None

    def _can_start(self, entry: _ReadyEntry) -> bool:
        _, _, _, item = entry
        return all(
            self._head(claim.resource) is entry
            and self._used[claim.resource] + claim.units
            <= self._resources[claim.resource].capacity
            for claim in item.claims
        )

    def _queue_candidate(self, entry: _ReadyEntry) -> None:
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
        settlements=None,
        **changes,
    ) -> None:
        run._transition(node_id, status, **changes)
        deferred = settlements is not None
        if self._journal is not None:
            if settlements is None:
                settlements = []
            settlements.append(run._results[node_id])
        run._unfinished -= 1
        if not propagate:
            if not deferred and settlements:
                self._record_settlements(run, settlements)
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
                    if self._journal is not None:
                        settlements.append(run._results[successor])
                    run._unfinished -= 1
                    completed.append(successor)
        if not deferred and settlements:
            self._record_settlements(run, settlements)

    def _record_settlements(self, run: WorkflowRun, results) -> None:
        if len(results) == 1:
            self._record_journal("_settle", run.run_id, results[0])
        else:
            self._record_journal("_settle_many", run.run_id, results)

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
        if self._journal is not None:
            self._record_journal("_finish", run.run_id, run._status)
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

        if self._journal_error is not None:
            return

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
            # Bind this dispatch before the loop advances to another node.
            context = ActionContext(
                run.run_id,
                node_id,
                lambda data, run=run, node_id=node_id: run._emit(
                    node_id, "feedback", data
                ),
                max_pending_feedback=self._max_pending_feedback,
                max_feedback_bytes=self._max_feedback_bytes,
                checkpoint=lambda data, run=run, node_id=node_id: self._checkpoint_node(
                    run, node_id, data
                ),
                feedback_overflow=lambda count, run=run, node_id=node_id: run._emit(
                    node_id, "feedback_overflow", {"dropped": count}
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
                self._workers_drained.clear()
                worker.add_done_callback(partial(self._worker_done, run, item, context))

    def _worker_done(
        self,
        run: WorkflowRun,
        item: CompiledNode,
        context: ActionContext,
        worker: asyncio.Future,
    ) -> None:
        self._workers.discard(worker)
        if not self._workers:
            self._workers_drained.set()
        if context.operation_id in self._draining:
            self._draining.remove(context.operation_id)
            context._finish()
            self._running_count -= 1
            run._emit(item.node.node_id, "adapter_drained")
            self._wake.set()
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
        # Keep every enclosing scope visible to wait-cycle checks. Weak entries
        # do not retain finished engines through long-lived child task contexts.
        execution_token = _current_execution.set(
            (
                *_current_execution.get(),
                (weakref.ref(self), weakref.ref(context)),
            )
        )
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
                lambda upstream_node_id: run._results[upstream_node_id].output,
            )
            if self._max_input_bytes is not None:
                inputs = _snapshot(inputs, max_bytes=self._max_input_bytes)
            action = self._actions[node.action]
            if action.input_schema is not None:
                inputs = action.input_schema.validate(inputs)
                if not isinstance(inputs, dict):
                    raise CommandValidationError(
                        "validated action inputs must be a dictionary"
                    )
                # Schema copies or overrides can change the prepared payload.
                inputs = _snapshot(inputs, max_bytes=self._max_input_bytes)
            # Recheck after input preparation: a delayed event-loop timer must
            # not allow a new device command to slip past an expired budget.
            self._check_run_deadline(run)
            if node_deadline is not None and self._loop.time() >= node_deadline:
                self._expire_node(run, node.node_id)
            context.raise_if_cancelled()
            if self._journal is not None:
                # Commit intent before effects; recheck the budget after fsync.
                await self._dispatch_barrier.commit(run, run._results[node.node_id])
                self._check_run_deadline(run)
                if node_deadline is not None and self._loop.time() >= node_deadline:
                    self._expire_node(run, node.node_id)
                context.raise_if_cancelled()
            context._handler_started = True
            output = _snapshot(
                await action.handler(context, inputs), max_bytes=self._max_output_bytes
            )
            if action.output_schema is not None:
                # RequestSchema allows None as an empty command for legacy APIs;
                # a declared object result requires an actual JSON object.
                if not isinstance(output, dict):
                    raise ActionOutputError("action output must be a dictionary")
                try:
                    validated_output = action.output_schema.validate(output)
                except CommandValidationError as exc:
                    raise ActionOutputError(str(exc)) from exc
                # Retain only the final JSON copy after schema validation succeeds.
                output = None
                if not isinstance(validated_output, dict):
                    raise ActionOutputError("action output must be a dictionary")
                output = _snapshot(
                    validated_output, max_bytes=self._max_output_bytes
                )
        except ActionCancelled as exc:
            status = NodeStatus.CANCELLED
            error = _error_text(exc)
        except asyncio.CancelledError as exc:
            context._request_cancel("adapter_interrupted")
            status = (
                NodeStatus.UNKNOWN if context._handler_started else NodeStatus.CANCELLED
            )
            error = _error_text(exc)
        except ExecutionUncertain as exc:
            status = NodeStatus.UNKNOWN
            error = _error_text(exc)
        except JournalError as exc:
            status = (
                NodeStatus.UNKNOWN if context._handler_started else NodeStatus.FAILED
            )
            error = _error_text(exc)
        except Exception as exc:
            status = NodeStatus.FAILED
            error = _error_text(exc)
        finally:
            if timer is not None:
                timer.cancel()
            _current_execution.reset(execution_token)

        if status != NodeStatus.SUCCEEDED and output is not None:
            # A failing custom schema can mutate its diagnostic data.
            try:
                output = _snapshot(output, max_bytes=self._max_output_bytes)
            except Exception:
                output = None

        # An adapter that monopolizes the loop can postpone timers; completion
        # still has to respect the elapsed budget. This cannot provide hard stop.
        self._check_run_deadline(run)
        if node_deadline is not None and self._loop.time() >= node_deadline:
            self._expire_node(run, node.node_id)
        if (
            context._stop_deadline is not None
            and self._loop.time() >= context._stop_deadline
        ):
            self._stop_confirmation_expired(run, node.node_id, context)
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
        draining: bool = False,
    ) -> None:
        node = item.node
        if run._results[node.node_id].status.terminal:
            return
        stop_timer = self._stop_timers.pop(context.operation_id, None)
        if stop_timer is not None:
            stop_timer.cancel()
        feedback_dropped = context._finish(draining=draining)
        if status != NodeStatus.UNKNOWN and context.cancel_requested:
            if context.cancel_reason in {"timeout", "workflow_timeout"}:
                status = NodeStatus.TIMED_OUT
            elif context.cancel_reason in {"scheduler_failed", "journal_failed"}:
                status = NodeStatus.FAILED
            else:
                status = NodeStatus.CANCELLED
            error = error or context.cancel_reason
        if status == NodeStatus.UNKNOWN:
            self._quarantined[context.operation_id] = (run, node.node_id)
        else:
            self._release(context.operation_id)
        if not draining:
            self._running_count -= 1
        self._complete_node(
            run,
            node.node_id,
            status,
            output=output,
            error=error,
            feedback_dropped=feedback_dropped,
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
