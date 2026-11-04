"""Durable concurrent admission crosses an all-or-nothing device I/O barrier."""

import asyncio
import sqlite3
import time
from unittest.mock import patch

import pytest

from orchestrion.workflow import (
    Action,
    FailurePolicy,
    JournalError,
    Node,
    NodeStatus,
    Resource,
    ResourceClaim,
    RunStatus,
    SQLiteJournal,
    Workflow,
    WorkflowEngine,
)


def test_concurrent_intents_are_committed_atomically_before_any_adapter_entry(tmp_path):
    async def scenario():
        path = tmp_path / "cell.db"
        entries, observations = [], []

        async def inspect(context, inputs):
            with sqlite3.connect(path) as reader:
                assert (
                    reader.execute(
                        "SELECT count(*) FROM nodes WHERE status='pending'"
                    ).fetchone()[0]
                    == 0
                )
            entries.append(context.operation_id)

        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("inspect", inspect)], journal=journal
            ) as engine:
                original = journal._event

                def observe(db, run_id, node_id, kind, detail=None, *, trim=True):
                    original(db, run_id, node_id, kind, detail, trim=trim)
                    if kind == "dispatch_intent":
                        with sqlite3.connect(path) as reader:
                            observations.append(
                                reader.execute(
                                    "SELECT count(*) FROM nodes WHERE lease_active=1"
                                ).fetchone()[0]
                            )

                with patch.object(journal, "_event", side_effect=observe):
                    runs = [
                        engine.submit(
                            Workflow(tuple(Node(str(i), "inspect") for i in range(4)))
                        )
                        for _ in range(2)
                    ]
                    results = await asyncio.gather(*(run.wait(2) for run in runs))
                assert observations == [0] * 8
                assert len(entries) == 8
                assert all(result.status == RunStatus.SUCCEEDED for result in results)

    asyncio.run(scenario())


@pytest.mark.parametrize("history_limit", [1, 2, 3])
def test_interleaved_dispatch_batches_keep_each_runs_event_limit(tmp_path, history_limit):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        with SQLiteJournal(
            tmp_path / "cell.db", max_events_per_run=history_limit
        ) as journal:
            async def inspect(context, inputs):
                events = journal.events(context.run_id)
                assert len(events) == history_limit
                assert all(event["run_id"] == context.run_id for event in events)
                assert all(event["kind"] == "dispatch_intent" for event in events)
                assert [event["node_id"] for event in events] == [
                    str(index) for index in range(6 - history_limit, 6)
                ]
                calls.append(context.operation_id)
                if len(calls) == 12:
                    entered.set()
                await asyncio.wait_for(release.wait(), 2)

            async with WorkflowEngine(
                [Action("inspect", inspect)], journal=journal
            ) as engine:
                runs = [
                    engine.submit(
                        Workflow(tuple(Node(str(index), "inspect") for index in range(6)))
                    )
                    for _ in range(2)
                ]
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                finally:
                    release.set()
                results = await asyncio.gather(*(run.wait(2) for run in runs))
                assert all(result.status == RunStatus.SUCCEEDED for result in results)
                for run in runs:
                    events = journal.events(run.run_id)
                    assert len(events) == history_limit
                    assert events[-1]["kind"] == "finished"
                    assert all(
                        event["kind"] in {"settled", "finished"} for event in events
                    )
                    assert journal.events(run.run_id, after=events[-1]["sequence"]) == []

    asyncio.run(scenario())


def test_fault_mid_dispatch_barrier_does_not_enter_any_adapter_or_keep_partial_intents(
    tmp_path,
):
    path = tmp_path / "cell.db"
    calls = []

    async def scenario():
        async def record(context, inputs):
            calls.append(context.operation_id)

        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("record", record, (ResourceClaim("arm"),))],
                [Resource("arm", capacity=8)],
                journal=journal,
            ) as engine:
                original = journal._event

                def fault(db, run_id, node_id, kind, detail=None, *, trim=True):
                    if kind == "dispatch_intent" and node_id == "3":
                        raise JournalError("disk full during group dispatch")
                    original(db, run_id, node_id, kind, detail, trim=trim)

                with patch.object(journal, "_event", side_effect=fault):
                    run = engine.submit(
                        Workflow(tuple(Node(str(i), "record") for i in range(8))),
                        idempotency_key="batch",
                    )
                    assert (await run.wait(2)).status == RunStatus.FAILED
                assert not calls
                assert not engine.resource_snapshot()["quarantined"]
                record = journal.get_run(run.run_id)
                assert all(node["status"] == "pending" for node in record["nodes"])
                assert all(not node["lease_active"] for node in record["nodes"])
                assert [e["kind"] for e in journal.events(run.run_id)] == ["admitted"]

    async def restart():
        with SQLiteJournal(path) as journal:
            async with WorkflowEngine([], [Resource("arm")], journal=journal) as engine:
                result = await engine.get_run_by_key("batch").wait(1)
                assert all(
                    node.status == NodeStatus.CANCELLED
                    for node in result.nodes.values()
                )
                assert not engine.resource_snapshot()["quarantined"]

    asyncio.run(scenario())
    asyncio.run(restart())


@pytest.mark.parametrize("interrupt", ["cancel_run", "cancel_worker"])
def test_cancellation_while_intent_waits_for_commit_has_no_device_effect(
    tmp_path, interrupt
):
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.operation_id)

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("record", record)], journal=journal
            ) as engine:
                original = engine._dispatch_barrier.commit

                async def cancel_before_flush(run, result):
                    cancel = (
                        run.cancel
                        if interrupt == "cancel_run"
                        else asyncio.current_task().cancel
                    )
                    asyncio.get_running_loop().call_soon(cancel)
                    await original(run, result)

                with patch.object(
                    engine._dispatch_barrier, "commit", side_effect=cancel_before_flush
                ):
                    run = engine.submit(Workflow((Node("n", "record"),)))
                    assert (await run.wait(1)).status == RunStatus.CANCELLED
                assert not calls
                assert not engine.resource_snapshot()["quarantined"]
                record = journal.get_run(run.run_id)
                assert record["nodes"][0]["status"] == "cancelled"
                assert not record["nodes"][0]["lease_active"]
                assert "dispatch_intent" not in [
                    e["kind"] for e in journal.events(run.run_id)
                ]

    asyncio.run(scenario())


@pytest.mark.parametrize("budget_scope", ["node", "workflow"])
def test_slow_group_commit_cannot_start_expired_commands(tmp_path, budget_scope):
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("record", record)], journal=journal
            ) as engine:
                original = journal._start_many

                def slow_commit(records):
                    original(records)
                    time.sleep(0.02)

                with patch.object(journal, "_start_many", side_effect=slow_commit):
                    run = engine.submit(
                        Workflow(
                            tuple(
                                Node(
                                    str(i),
                                    "record",
                                    timeout=0.005 if budget_scope == "node" else None,
                                )
                                for i in range(4)
                            ),
                            timeout=0.005 if budget_scope == "workflow" else None,
                            failure_policy=FailurePolicy.CONTINUE_INDEPENDENT,
                        )
                    )
                    result = await run.wait(2)
                assert not calls
                assert all(
                    node.status == NodeStatus.TIMED_OUT
                    for node in result.nodes.values()
                )
                assert not engine.resource_snapshot()["quarantined"]
                assert all(
                    not n["lease_active"] for n in journal.get_run(run.run_id)["nodes"]
                )

    asyncio.run(scenario())
