"""Workflow run handles, defensive snapshots and bounded in-memory events."""

import asyncio
import time
import uuid
from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .actions import ActionContext
from .compiler import CompiledNode
from .diagnostics import explain_node
from .model import (
    FailurePolicy,
    NodeResult,
    NodeStatus,
    RunResult,
    RunStatus,
    Workflow,
    WorkflowEvent,
    _name,
    _operation_id,
    _snapshot,
    _timeout,
)

if TYPE_CHECKING:
    from .engine import WorkflowEngine


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
            item.node.node_id: NodeResult(
                item.node.node_id, item.node.action, action_version=item.action_version
            )
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

    @classmethod
    def _restore(
        cls, engine: "WorkflowEngine", record: Dict[str, Any]
    ) -> "WorkflowRun":
        """Restore a final handle, never a graph that can dispatch device work."""
        run = cls.__new__(cls)
        run._engine = engine
        run._run_id = record["run_id"]
        run._idempotency_key = record["idempotency_key"]
        run._deadline = run._deadline_timer = None
        run._deadline_expired = False
        run._by_id = {}
        run._remaining_dependencies = {}
        run._successors = {}
        run._unfinished = 0
        run._policy = FailurePolicy.FAIL_FAST
        run._results = {
            node["node_id"]: NodeResult(
                node_id=node["node_id"],
                action=node["action"],
                status=NodeStatus(node["status"]),
                output=node["output"],
                error=node["error"],
                started_at=node["started_at"],
                finished_at=node["finished_at"],
                feedback_dropped=node["feedback_dropped"],
                checkpoint=node["checkpoint"],
                action_version=node["action_version"],
            )
            for node in record["nodes"]
        }
        run._contexts = {}
        run._events = deque(maxlen=engine._max_events)
        run._sequence = 0
        run._cancelled = run._failed_fast = False
        run._status = RunStatus(record["status"])
        run._done = engine._loop.create_future()
        run._done.set_result(None)
        for node_id in run._results:
            run._emit(node_id, "recovered")
        return run

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def idempotency_key(self) -> Optional[str]:
        return self._idempotency_key

    def operation_id(self, node_id: str) -> str:
        """Return the stable encoded ID used by adapters and reconciliation."""
        _name(node_id, "node ID")
        if node_id not in self._results:
            raise KeyError(node_id)
        return _operation_id(self.run_id, node_id)

    @property
    def done(self) -> bool:
        return self._done.done()

    def snapshot(self) -> RunResult:
        """Return a defensive snapshot; terminal UNKNOWN is not a stop confirmation."""
        # Result metadata is immutable. Copy only the JSON payloads instead of
        # reconstructing every field and enum through generic deepcopy machinery.
        results = {
            node_id: replace(
                result,
                output=_snapshot(result.output),
                checkpoint=_snapshot(result.checkpoint),
            )
            for node_id, result in self._results.items()
        }
        return RunResult(self.run_id, self._status, results)

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
            replace(event, detail=_snapshot(event.detail))
            for event in self._events
            if event.sequence > after
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
        """Record JSON owned by the engine; public reads return isolated copies."""
        self._sequence += 1
        self._events.append(
            WorkflowEvent(
                self._sequence,
                self.run_id,
                node_id,
                kind,
                self._results[node_id].status,
                time.time(),
                detail,
            )
        )

    def _transition(self, node_id: str, status: NodeStatus, **changes) -> None:
        if status.terminal:
            changes["finished_at"] = time.time()
        self._results[node_id] = replace(
            self._results[node_id], status=status, **changes
        )
        self._emit(node_id, "state", changes.get("error"))
