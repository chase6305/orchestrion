"""On-demand scheduler diagnostics; no device calls or state transitions."""

from typing import TYPE_CHECKING, Any, Dict, Optional

from .model import NodeStatus

if TYPE_CHECKING:
    from .engine import WorkflowEngine, _ReadyEntry
    from .run import WorkflowRun


def _queue_head(engine: "WorkflowEngine", resource: str) -> "Optional[_ReadyEntry]":
    # Reading diagnostics must not compact queues or alter scheduling order.
    waiters = engine._resource_waiters[resource]
    if waiters and engine._is_ready(waiters[0]):
        return waiters[0]
    return min(
        (entry for entry in waiters if engine._is_ready(entry)),
        default=None,
    )


def _owners(engine: "WorkflowEngine") -> Dict[str, list]:
    owners = {name: [] for name in engine._resources}
    for operation_id, claims in engine._leases.items():
        for claim in claims:
            owners[claim.resource].append(
                {
                    "operation_id": operation_id,
                    "units": claim.units,
                    "quarantined": operation_id in engine._quarantined,
                }
            )
    return owners


def explain_node(run: "WorkflowRun", node_id: str) -> Dict[str, Any]:
    engine = run._engine
    result = run._results[node_id]
    operation_id = run.operation_id(node_id)
    blockers = []
    if result.status == NodeStatus.PENDING:
        blockers.append(
            {
                "kind": "dependencies",
                "nodes": [
                    dependency
                    for dependency in run._by_id[node_id].dependencies
                    if run._results[dependency].status != NodeStatus.SUCCEEDED
                ],
            }
        )
    elif result.status == NodeStatus.READY:
        if engine._running_count >= engine._max_concurrency:
            blockers.append(
                {
                    "kind": "concurrency",
                    "running": engine._running_count,
                    "limit": engine._max_concurrency,
                }
            )
        owners = _owners(engine)
        for claim in run._by_id[node_id].claims:
            available = (
                engine._resources[claim.resource].capacity
                - engine._used[claim.resource]
            )
            if available < claim.units:
                blockers.append(
                    {
                        "kind": "resource_capacity",
                        "resource": claim.resource,
                        "required": claim.units,
                        "available": available,
                        "owners": owners[claim.resource],
                    }
                )
            head = _queue_head(engine, claim.resource)
            if head is not None:
                _, _, head_run, head_node = head
                if head_run is not run or head_node.node.node_id != node_id:
                    blockers.append(
                        {
                            "kind": "resource_queue",
                            "resource": claim.resource,
                            "ahead": head_run.operation_id(head_node.node.node_id),
                        }
                    )
        if not blockers:
            blockers.append({"kind": "dispatch"})
    elif result.status == NodeStatus.CANCELLING:
        blockers.append(
            {
                "kind": "stop_confirmation",
                "reason": run._contexts[node_id].cancel_reason,
            }
        )
    elif operation_id in engine._quarantined:
        blockers.append({"kind": "quarantine", "operation_id": operation_id})
    context = run._contexts.get(node_id)
    return {
        "run_id": run.run_id,
        "node_id": node_id,
        "action": result.action,
        "status": result.status.value,
        "blockers": blockers,
        "feedback": (
            {"pending": 0, "dropped": result.feedback_dropped}
            if context is None
            else context._feedback_snapshot()
        ),
    }


def scheduler_snapshot(engine: "WorkflowEngine") -> Dict[str, Any]:
    snapshot = engine.resource_snapshot()
    owners = _owners(engine)
    for name, resource in snapshot["resources"].items():
        resource["owners"] = owners[name]
        resource["waiting"] = sum(
            engine._is_ready(entry) for entry in engine._resource_waiters[name]
        )
        head = _queue_head(engine, name)
        if head is None:
            resource["next_operation"] = None
        else:
            _, _, head_run, head_node = head
            resource["next_operation"] = head_run.operation_id(head_node.node.node_id)
    counts = dict.fromkeys((status.value for status in NodeStatus), 0)
    for run in engine._runs.values():
        for result in run._results.values():
            counts[result.status.value] += 1
    uncertain = {run.run_id for run, _ in engine._quarantined.values()}
    snapshot.update(
        {
            "closed": engine._closed,
            "active_runs": len(engine._runs),
            "max_active_runs": engine._max_active_runs,
            "running_actions": engine._running_count,
            "draining_adapters": len(engine._draining),
            "stop_timeout": engine._stop_timeout,
            "max_concurrency": engine._max_concurrency,
            "max_pending_feedback_per_action": engine._max_pending_feedback,
            "scheduler_error": engine._driver_error,
            "journal_error": engine._journal_error,
            "durable": engine._journal is not None,
            "journal_mode": None
            if engine._journal is None
            else engine._journal.journal_mode,
            "payload_limits": {
                "input_per_node": engine._max_input_bytes,
                "output_per_node": engine._max_output_bytes,
                "feedback_per_report": engine._max_feedback_bytes,
            },
            "nodes": counts,
            "idempotency_keys": {
                "retained": len(engine._submissions),
                "limit": engine._max_idempotency_keys,
                "prunable": sum(
                    run.done and run.run_id not in uncertain
                    for _, run in engine._submissions.values()
                ),
            },
        }
    )
    return snapshot
