"""Rebuild retained handles and quarantine before admitting new device commands."""

from typing import TYPE_CHECKING

from .compiler import check_claims
from .journal import JournalError, _run_status
from .model import NodeStatus, ResourceClaim, RunStatus
from .run import WorkflowRun

if TYPE_CHECKING:
    from .engine import WorkflowEngine


def restore_journal(engine: "WorkflowEngine") -> None:
    journal = engine._journal
    records = journal._load()
    used = dict.fromkeys(engine._resources, 0)
    for record in records:
        run_status = RunStatus(record["status"])
        for node in record["nodes"]:
            status = NodeStatus(node["status"])
            if record["finished_at"] is not None and not status.terminal:
                raise JournalError("finished durable run has an unfinished operation")
            if (
                status in {NodeStatus.RUNNING, NodeStatus.CANCELLING}
                and not node["lease_active"]
            ):
                raise JournalError("dispatched durable operation is missing its lease")
            reconciled = (
                node["reconciled_at"] is not None
                and isinstance(node["reconciliation_evidence"], str)
                and bool(node["reconciliation_evidence"].strip())
            )
            if (
                status == NodeStatus.UNKNOWN
                and not node["lease_active"]
                and not reconciled
            ):
                raise JournalError(
                    "unknown durable operation has no reconciliation evidence"
                )
            if (
                node["reconciled_at"] is not None
                or node["reconciliation_evidence"] is not None
            ):
                if (
                    status != NodeStatus.UNKNOWN
                    or node["lease_active"]
                    or not reconciled
                ):
                    raise JournalError(
                        "durable reconciliation has an inconsistent operation state"
                    )
            if node["lease_active"]:
                if status not in {
                    NodeStatus.RUNNING,
                    NodeStatus.CANCELLING,
                    NodeStatus.UNKNOWN,
                }:
                    raise JournalError(
                        "durable lease has an inconsistent operation state"
                    )
                claims = tuple(
                    ResourceClaim(name, units) for name, units in node["claims"].items()
                )
                check_claims(claims, engine._resources)
                for claim in claims:
                    used[claim.resource] += claim.units
        if record["finished_at"] is not None and run_status.value != _run_status(
            record["nodes"]
        ):
            raise JournalError("durable run outcome disagrees with its operations")
    for name, units in used.items():
        if units > engine._resources[name].capacity:
            raise ValueError("recovered leases exceed resource capacity: " + name)
    # Do not mutate the database until every unresolved claim is feasible.
    journal._recover(records)
    restored = [(record, WorkflowRun._restore(engine, record)) for record in records]
    submissions, leases, quarantined = {}, {}, {}
    for record, run in restored:
        if run.idempotency_key is not None:
            submissions[run.idempotency_key] = (record["fingerprint"], run)
        for node in record["nodes"]:
            if node["lease_active"]:
                operation_id = run.operation_id(node["node_id"])
                leases[operation_id] = tuple(
                    ResourceClaim(name, units) for name, units in node["claims"].items()
                )
                quarantined[operation_id] = (run, node["node_id"])
    # Publish the fully reconstructed ownership map as one event-loop turn.
    engine._submissions.update(submissions)
    engine._leases.update(leases)
    engine._quarantined.update(quarantined)
    engine._used.update(used)
