"""Crash recovery, durable admission, and conservative device ownership."""

import asyncio
import json
import sqlite3
import subprocess
import sys
from dataclasses import replace
from unittest.mock import patch

import pytest

from orchestrion.workflow import (
    Action,
    ExecutionUncertain,
    JournalError,
    JournalInUse,
    Node,
    NodeStatus,
    Resource,
    ResourceClaim,
    RunStatus,
    SQLiteJournal,
    Workflow,
    WorkflowEngine,
)


async def echo(context, inputs):
    return inputs


def test_completed_submission_replays_across_restart_and_prunes_durably(tmp_path):
    path = tmp_path / "cell.db"
    workflow = Workflow((Node("read", "echo", {"value": [42]}),))

    async def first():
        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("echo", echo)], journal=journal
            ) as engine:
                run = engine.submit(workflow, idempotency_key="cycle")
                assert (await run.wait(1)).status == RunStatus.SUCCEEDED
                record = journal.get_run(run.run_id)
                assert record["nodes"][0]["output"] == {"value": [42]}
                record["nodes"][0]["output"]["value"].clear()
                assert journal.get_run(run.run_id)["nodes"][0]["output"] == {
                    "value": [42]
                }
                return run.run_id

    run_id = asyncio.run(first())

    async def restart():
        calls = []

        async def must_not_repeat(context, inputs):
            calls.append(context.operation_id)
            return inputs

        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("echo", must_not_repeat)], journal=journal
            ) as engine:
                run = engine.get_run_by_key("cycle")
                assert run.done and run.run_id == run_id
                assert engine.submit(workflow, idempotency_key="cycle") is run
                assert (await run.wait(1)).nodes["read"].output == {"value": [42]}
                assert not calls
                with pytest.raises(ValueError, match="different workflow"):
                    engine.submit(
                        replace(workflow, nodes=(Node("read", "echo", {"value": 43}),)),
                        idempotency_key="cycle",
                    )
                assert journal.prune_completed_runs(0) == 0
                assert engine.prune_idempotency_keys(0) == 1
                assert journal.get_run(run_id)["idempotency_key"] is None
                assert journal.prune_completed_runs(0) == 1
                assert journal.get_run(run_id) is None

    asyncio.run(restart())
    with SQLiteJournal(path) as journal:
        assert journal._load() == []


_CRASH_SCRIPT = """
import asyncio, json, os, sys
from orchestrion.workflow import (
    Action, Node, Resource, ResourceClaim, SQLiteJournal, Workflow, WorkflowEngine,
)
async def prepare(context, inputs):
    return {"ready": True}
async def crash(context, inputs):
    print(json.dumps({"run": context.run_id, "operation": context.operation_id}), flush=True)
    os._exit(73)
async def main():
    with SQLiteJournal(sys.argv[1]) as journal:
        async with WorkflowEngine([
            Action("prepare", prepare),
            Action("move", crash, (ResourceClaim("arm"),)),
        ], [Resource("arm")], journal=journal) as engine:
            run = engine.submit(Workflow((
                Node("prepare", "prepare"),
                Node("move", "move", after=("prepare",)),
                Node("after", "prepare", after=("move",)),
            )), idempotency_key="interrupted-cycle")
            await run.wait()
asyncio.run(main())
"""


def crash_device_process(path):
    result = subprocess.run(
        [sys.executable, "-c", _CRASH_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 73, result.stderr
    return json.loads(result.stdout)


def test_hard_process_exit_restores_unknown_and_prevents_device_redispatch(tmp_path):
    path = tmp_path / "cell.db"
    interrupted = crash_device_process(path)

    async def scenario():
        calls = []

        async def move(context, inputs):
            calls.append(context.operation_id)
            return {"stopped": True}

        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("move", move, (ResourceClaim("arm"),))],
                [Resource("arm")],
                journal=journal,
            ) as engine:
                recovered = engine.get_run_by_key("interrupted-cycle")
                result = await recovered.wait(1)
                assert recovered.run_id == interrupted["run"]
                assert result.status == RunStatus.NEEDS_ATTENTION
                assert result.nodes["prepare"].status == NodeStatus.SUCCEEDED
                assert result.nodes["prepare"].output == {"ready": True}
                assert result.nodes["move"].status == NodeStatus.UNKNOWN
                assert result.nodes["after"].status == NodeStatus.CANCELLED
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
                assert recovered.explain("move")["blockers"][0]["kind"] == "quarantine"
                assert engine.prune_idempotency_keys(0) == 0
                assert journal.prune_completed_runs(0) == 0
                next_run = engine.submit(Workflow((Node("next", "move"),)))
                with pytest.raises(asyncio.TimeoutError):
                    await next_run.wait(0.01)
                assert calls == []
                engine.confirm_quiescent(
                    interrupted["operation"], evidence="controller reports stationary"
                )
                assert (await next_run.wait(1)).status == RunStatus.SUCCEEDED
                assert len(calls) == 1
                events = journal.events(interrupted["run"])
                assert events[-1]["kind"] == "reconciled"
                assert events[-1]["detail"] == {
                    "evidence": "controller reports stationary"
                }

    asyncio.run(scenario())

    # Reconciliation is durable, while the historical outcome remains UNKNOWN.
    async def restart_again():
        with SQLiteJournal(path) as journal:
            async with WorkflowEngine([], [Resource("arm")], journal=journal) as engine:
                run = engine.get_run_by_key("interrupted-cycle")
                assert run.snapshot().status == RunStatus.NEEDS_ATTENTION
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
                assert engine.prune_idempotency_keys(0) == 1

    asyncio.run(restart_again())


def test_intent_is_committed_before_device_io_and_completion_before_wait_returns(
    tmp_path,
):
    path = tmp_path / "cell.db"

    async def scenario():
        async def inspect(context, inputs):
            # An independent connection sees only committed data.
            with sqlite3.connect(path) as db:
                row = db.execute(
                    "SELECT status,lease_active FROM nodes WHERE node_id='move'"
                ).fetchone()
                assert row == ("running", 1)
            return {"done": True}

        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("inspect", inspect)], journal=journal
            ) as engine:
                run = engine.submit(Workflow((Node("move", "inspect"),)))
                await run.wait(1)
                record = journal.get_run(run.run_id)
                assert record["status"] == "succeeded"
                assert record["finished_at"] is not None
                assert not record["nodes"][0]["lease_active"]

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["_admit", "_start", "_settle", "_finish"])
@pytest.mark.parametrize("error_type", [JournalError, asyncio.CancelledError])
def test_journal_failure_closes_admission_and_preserves_conservative_recovery(
    tmp_path, stage, error_type
):
    path = tmp_path / "cell.db"
    calls = []

    async def scenario():
        async def record(context, inputs):
            calls.append(context.operation_id)
            return {"finished": True}

        with SQLiteJournal(path) as journal:
            engine = WorkflowEngine([Action("record", record)], journal=journal)
            async with engine:
                with patch.object(journal, stage, side_effect=error_type("disk full")):
                    if stage == "_admit":
                        with pytest.raises(JournalError, match="disk full"):
                            engine.submit(Workflow((Node("n", "record"),)))
                    else:
                        run = engine.submit(
                            Workflow((Node("n", "record"),)), idempotency_key="key"
                        )
                        await run.wait(1)
                    assert engine.scheduler_snapshot()["journal_error"] is not None
                    with pytest.raises(RuntimeError, match="closed"):
                        engine.submit(Workflow((Node("another", "record"),)))
            assert not engine._leases

    asyncio.run(scenario())
    assert len(calls) == int(stage in {"_settle", "_finish"})

    async def restart():
        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("record", echo)], journal=journal
            ) as engine:
                restored = engine.get_run_by_key("key")
                if stage == "_admit":
                    assert restored is None
                else:
                    status = restored.snapshot().nodes["n"].status
                    assert (
                        status
                        == {
                            "_start": NodeStatus.CANCELLED,
                            "_settle": NodeStatus.UNKNOWN,
                            "_finish": NodeStatus.SUCCEEDED,
                        }[stage]
                    )
                    assert len(engine.resource_snapshot()["quarantined"]) == int(
                        stage == "_settle"
                    )

    asyncio.run(restart())


@pytest.mark.parametrize("queued_count", [0, 16])
def test_failed_intent_stops_other_active_adapters_before_queued_device_io(
    tmp_path, queued_count
):
    async def scenario():
        entered = asyncio.Event()
        stopped = asyncio.Event()
        calls = []

        async def hold(context, inputs):
            entered.set()
            assert await context.wait_cancelled() == "journal_failed"
            stopped.set()
            context.raise_if_cancelled()

        async def never(context, inputs):
            calls.append(context.node_id)

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [
                    Action("hold", hold, (ResourceClaim("arm"),)),
                    Action("never", never),
                ],
                [Resource("arm")],
                max_concurrency=2,
                journal=journal,
            ) as engine:
                holder = engine.submit(Workflow((Node("hold", "hold"),)))
                await asyncio.wait_for(entered.wait(), 1)
                with patch.object(
                    journal, "_start", side_effect=JournalError("offline")
                ):
                    blocked = engine.submit(Workflow((Node("blocked", "never"),)))
                    queued = [
                        engine.submit(
                            Workflow(
                                (
                                    Node(
                                        "queued",
                                        "never",
                                        claims=(
                                            (ResourceClaim("arm"),) if index % 2 else ()
                                        ),
                                    ),
                                )
                            )
                        )
                        for index in range(queued_count)
                    ]
                    assert (await blocked.wait(1)).status == RunStatus.FAILED
                    for pending in queued:
                        result = await pending.wait(1)
                        assert result.status == RunStatus.FAILED
                        assert result.nodes["queued"].started_at is None
                    assert (await holder.wait(1)).status == RunStatus.FAILED
                assert stopped.is_set() and calls == []
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0

    asyncio.run(scenario())


def test_owner_lock_survives_commits_and_releases_after_close(tmp_path):
    path = tmp_path / "cell.db"
    with SQLiteJournal(path):
        with pytest.raises(JournalInUse):
            SQLiteJournal(path)
        with pytest.raises(JournalInUse):
            SQLiteJournal(tmp_path / "." / "cell.db")
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from orchestrion.workflow import SQLiteJournal, JournalInUse
try:
    SQLiteJournal(sys.argv[1])
except JournalInUse:
    sys.exit(17)
""",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert probe.returncode == 17, probe.stderr
    with SQLiteJournal(path) as journal:
        assert journal._load() == []


def test_journal_rejects_closure_and_rebinding_until_engine_drains(tmp_path):
    async def scenario():
        journal = SQLiteJournal(tmp_path / "cell.db")
        engine = WorkflowEngine([Action("echo", echo)], journal=journal)
        try:
            with pytest.raises(JournalInUse, match="drain"):
                journal.close()
            with pytest.raises(JournalInUse, match="bound"):
                WorkflowEngine([], journal=journal)
            async with engine:
                await engine.submit(Workflow((Node("echo", "echo"),))).wait(1)
        finally:
            await engine.aclose(1)
            journal.close()
        journal.close()
        with pytest.raises(JournalError, match="closed"):
            journal.get_run("missing")

    asyncio.run(scenario())


def test_recovery_refuses_missing_or_insufficient_resource_configuration(tmp_path):
    path = tmp_path / "cell.db"
    crash_device_process(path)

    async def scenario():
        with SQLiteJournal(path) as journal:
            engine = WorkflowEngine([], journal=journal)
            with pytest.raises(ValueError, match="unknown resource"):
                await engine.__aenter__()
            assert engine.scheduler_snapshot()["journal_error"] is not None
        # Failed validation did not mutate the interrupted dispatch record.
        with SQLiteJournal(path) as journal:
            record = journal._load()[0]
            assert record["finished_at"] is None
            assert record["nodes"][1]["status"] == "running"

    asyncio.run(scenario())


def test_unknown_unkeyed_operation_is_restored_and_cannot_be_pruned(tmp_path):
    path = tmp_path / "cell.db"

    async def first():
        async def lost(context, inputs):
            raise ExecutionUncertain("disconnected")

        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("lost", lost, (ResourceClaim("arm"),))],
                [Resource("arm")],
                journal=journal,
            ) as engine:
                run = engine.submit(Workflow((Node("lost", "lost"),)))
                await run.wait(1)
                assert journal.prune_completed_runs(0) == 0
                return run.run_id

    run_id = asyncio.run(first())

    async def restart():
        with SQLiteJournal(path) as journal:
            async with WorkflowEngine([], [Resource("arm")], journal=journal) as engine:
                assert engine.resource_snapshot()["quarantined"][run_id + "/lost"]
                engine.confirm_quiescent(run_id + "/lost", evidence="manual inspection")
                assert journal.prune_completed_runs(0) == 1

    asyncio.run(restart())


def test_reconciliation_failure_keeps_quarantine_and_blocks_new_admission(tmp_path):
    async def scenario():
        async def lost(context, inputs):
            raise ExecutionUncertain("lost acknowledgement")

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("lost", lost, (ResourceClaim("arm"),))],
                [Resource("arm")],
                journal=journal,
            ) as engine:
                run = engine.submit(Workflow((Node("lost", "lost"),)))
                await run.wait(1)
                with patch.object(
                    journal, "_reconcile", side_effect=JournalError("disk")
                ):
                    with pytest.raises(JournalError):
                        engine.confirm_quiescent(
                            run.run_id + "/lost", evidence="stopped"
                        )
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
                assert run.explain("lost")["blockers"][0]["kind"] == "quarantine"

    asyncio.run(scenario())


def test_durable_audit_is_bounded_and_progress_does_not_force_database_commits(
    tmp_path,
):
    async def scenario():
        async def report(context, inputs):
            for index in range(200):
                context.report({"progress": index})
            return inputs

        with SQLiteJournal(tmp_path / "cell.db", max_events_per_run=3) as journal:
            async with WorkflowEngine(
                [Action("report", report)], journal=journal
            ) as engine:
                run = engine.submit(
                    Workflow(tuple(Node(str(i), "report") for i in range(8)))
                )
                await run.wait(1)
                events = journal.events(run.run_id)
                assert len(events) == 3
                assert events[-1]["kind"] == "finished"
                assert all(event["kind"] != "feedback" for event in events)
                assert (
                    journal.events(run.run_id, after=events[-2]["sequence"])
                    == events[-1:]
                )

    asyncio.run(scenario())


def test_unrelated_and_future_schema_databases_are_rejected_without_modification(
    tmp_path,
):
    path = tmp_path / "other.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated(value)")
        db.execute("INSERT INTO unrelated VALUES(42)")
    with pytest.raises(JournalError, match="unsupported"):
        SQLiteJournal(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM unrelated").fetchall() == [(42,)]
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    db.close()
    future = tmp_path / "future.db"
    with SQLiteJournal(future):
        pass
    with sqlite3.connect(future) as db:
        db.execute("PRAGMA user_version=99")
    db.close()
    with pytest.raises(JournalError, match="unsupported"):
        SQLiteJournal(future)
    with sqlite3.connect(future) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 99
    db.close()
    with pytest.raises(ValueError, match="filesystem"):
        SQLiteJournal(":memory:")


@pytest.mark.parametrize("budget_scope", ["node", "workflow"])
def test_slow_commit_cannot_dispatch_an_expired_device_command(tmp_path, budget_scope):
    import time

    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.operation_id)

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            original = journal._start

            def slow_commit(*args):
                original(*args)
                time.sleep(0.02)

            async with WorkflowEngine(
                [Action("record", record)], journal=journal
            ) as engine:
                workflow = Workflow(
                    (
                        Node(
                            "n",
                            "record",
                            timeout=0.005 if budget_scope == "node" else None,
                        ),
                    ),
                    timeout=0.005 if budget_scope == "workflow" else None,
                )
                with patch.object(journal, "_start", side_effect=slow_commit):
                    run = engine.submit(workflow)
                    result = await run.wait(1)
                assert result.nodes["n"].status == NodeStatus.TIMED_OUT
                assert calls == []
                assert not engine.resource_snapshot()["quarantined"]
                assert not journal.get_run(run.run_id)["nodes"][0]["lease_active"]

    asyncio.run(scenario())


def test_checkpoint_and_result_survive_hard_exit_after_durable_metadata(tmp_path):
    path = tmp_path / "cell.db"
    script = _CRASH_SCRIPT.replace(
        'print(json.dumps({"run": context.run_id, "operation": context.operation_id}), flush=True)',
        'context.checkpoint({"ticket": "remote-42"})\n    print(json.dumps({"run": context.run_id, "operation": context.operation_id}), flush=True)',
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 73, completed.stderr
    run_id = json.loads(completed.stdout)["run"]

    async def scenario():
        with SQLiteJournal(path) as journal:
            async with WorkflowEngine([], [Resource("arm")], journal=journal) as engine:
                result = await engine.get_run_by_key("interrupted-cycle").wait(1)
                assert result.run_id == run_id
                assert result.nodes["move"].checkpoint == {"ticket": "remote-42"}
                result.nodes["move"].checkpoint.clear()
                assert engine.get_run_by_key("interrupted-cycle").snapshot().nodes[
                    "move"
                ].checkpoint == {"ticket": "remote-42"}

    asyncio.run(scenario())


@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"), reason="requires eager task support"
)
def test_eager_dispatch_observes_committed_key_and_checkpoint_before_device_io(
    tmp_path,
):
    async def scenario():
        loop = asyncio.get_running_loop()
        original = loop.get_task_factory()
        with SQLiteJournal(tmp_path / "cell.db") as journal:

            async def immediate(context, inputs):
                run = engine.get_run_by_key("key")
                assert run.run_id == context.run_id
                assert journal.get_run(context.run_id)["idempotency_key"] == "key"
                context.checkpoint({"ticket": "42"})
                assert journal.get_run(context.run_id)["nodes"][0]["checkpoint"] == {
                    "ticket": "42"
                }
                return inputs

            async with WorkflowEngine(
                [Action("immediate", immediate)], journal=journal
            ) as engine:
                loop.set_task_factory(asyncio.eager_task_factory)
                try:
                    run = engine.submit(
                        Workflow((Node("n", "immediate"),)), idempotency_key="key"
                    )
                    assert (await run.wait(1)).status == RunStatus.SUCCEEDED
                    assert (
                        engine.submit(
                            Workflow((Node("n", "immediate"),)), idempotency_key="key"
                        )
                        is run
                    )
                finally:
                    loop.set_task_factory(original)

    asyncio.run(scenario())


def test_uninitialized_durable_snapshot_does_not_claim_resources_are_free(tmp_path):
    path = tmp_path / "cell.db"
    crash_device_process(path)
    with SQLiteJournal(path) as journal:
        engine = WorkflowEngine([], [Resource("arm")], journal=journal)
        snapshot = engine.resource_snapshot()
        assert snapshot["recovery_pending"]
        assert snapshot["resources"]["arm"]["used"] is None

        async def scenario():
            async with engine:
                snapshot = engine.resource_snapshot()
                assert not snapshot["recovery_pending"]
                assert snapshot["resources"]["arm"]["used"] == 1

        asyncio.run(scenario())


@pytest.mark.parametrize("failure_at", ["driver", "ready_queue"])
def test_scheduler_start_failure_releases_idempotency_key_on_disk_and_in_memory(
    tmp_path,
    monkeypatch,
    failure_at,
):
    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()

        def refuse(loop, coroutine, **kwargs):
            if coroutine.cr_code.co_name == "_schedule":
                raise RuntimeError("scheduler startup failed")
            return asyncio.Task(coroutine, loop=loop, **kwargs)

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("echo", echo)], journal=journal
            ) as engine:
                workflow = Workflow((Node("n", "echo"),))
                prepare = engine._make_ready

                def refuse_ready(run, item):
                    prepare(run, item)
                    raise RuntimeError("scheduler startup failed")

                with monkeypatch.context() as failure:
                    if failure_at == "driver":
                        loop.set_task_factory(refuse)
                    else:
                        failure.setattr(engine, "_make_ready", refuse_ready)
                    try:
                        with pytest.raises(RuntimeError, match="startup failed"):
                            engine.submit(workflow, idempotency_key="key")
                    finally:
                        loop.set_task_factory(original_factory)
                assert engine.get_run_by_key("key") is None
                assert not journal._load()
                good = engine.submit(workflow, idempotency_key="key")
                assert (await good.wait(1)).status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


def test_missing_dispatch_record_closes_admission_before_device_io(tmp_path):
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.operation_id)

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("record", record)], journal=journal
            ) as engine:
                run = engine.submit(Workflow((Node("n", "record"),)))
                # Simulate storage damage between admission and worker startup.
                with journal._db:
                    journal._db.execute(
                        "DELETE FROM nodes WHERE run_id=?", (run.run_id,)
                    )
                result = await run.wait(1)
                assert result.status == RunStatus.FAILED
                assert calls == []
                assert "missing durable" in engine.scheduler_snapshot()["journal_error"]

    asyncio.run(scenario())


def test_reconciliation_proof_survives_audit_event_eviction(tmp_path):
    async def scenario():
        async def lost(context, inputs):
            raise ExecutionUncertain("disconnected")

        with SQLiteJournal(tmp_path / "cell.db", max_events_per_run=1) as journal:
            async with WorkflowEngine(
                [Action("lost", lost)], journal=journal
            ) as engine:
                from orchestrion.workflow import FailurePolicy

                run = engine.submit(
                    Workflow(
                        (Node("a", "lost"), Node("b", "lost")),
                        FailurePolicy.CONTINUE_INDEPENDENT,
                    )
                )
                await run.wait(1)
                engine.confirm_quiescent(
                    run.run_id + "/a", evidence="controller A stationary"
                )
                engine.confirm_quiescent(
                    run.run_id + "/b", evidence="controller B stationary"
                )
                events = journal.events(run.run_id)
                assert len(events) == 1 and events[0]["node_id"] == "b"
                nodes = journal.get_run(run.run_id)["nodes"]
                assert nodes[0]["reconciliation_evidence"] == "controller A stationary"
                assert nodes[1]["reconciliation_evidence"] == "controller B stationary"
                assert all(node["reconciled_at"] is not None for node in nodes)
                assert all(not node["lease_active"] for node in nodes)

    asyncio.run(scenario())


def test_changed_action_version_rejects_durable_key_reuse(tmp_path):
    path = tmp_path / "cell.db"
    workflow = Workflow((Node("n", "echo"),))

    async def first():
        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("echo", echo, version="v1")], journal=journal
            ) as engine:
                run = engine.submit(workflow, idempotency_key="key")
                assert (await run.wait(1)).nodes["n"].action_version == "v1"
                assert journal.get_run(run.run_id)["nodes"][0]["action_version"] == "v1"

    asyncio.run(first())

    async def restart():
        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("echo", echo, version="v2")], journal=journal
            ) as engine:
                assert engine.describe()["actions"]["echo"]["version"] == "v2"
                with pytest.raises(ValueError, match="different workflow"):
                    engine.submit(workflow, idempotency_key="key")
                old = engine.get_run_by_key("key")
                assert old.snapshot().nodes["n"].action_version == "v1"

    asyncio.run(restart())


def test_forked_child_cannot_use_inherited_journal_connection(tmp_path):
    import os

    if not hasattr(os, "fork"):
        pytest.skip("requires fork")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os, sys
from orchestrion.workflow import SQLiteJournal, JournalInUse
with SQLiteJournal(sys.argv[1]) as journal:
    child = os.fork()
    if child == 0:
        try:
            journal.get_run("missing")
        except JournalInUse:
            os._exit(17)
        os._exit(1)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 17
    assert journal.get_run("missing") is None
""",
            str(tmp_path / "cell.db"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_encoded_operation_ids_restore_quarantine_and_diagnostics_consistently(
    tmp_path,
):
    async def first():
        async def lost(context, inputs):
            raise ExecutionUncertain("lost response")

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("lost", lost, (ResourceClaim("arm"),))],
                [Resource("arm")],
                journal=journal,
            ) as engine:
                run = engine.submit(
                    Workflow((Node("arm/move", "lost"),)), idempotency_key="key"
                )
                assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
                operation = run.operation_id("arm/move")
                assert operation in engine.resource_snapshot()["quarantined"]
                assert run.explain("arm/move")["blockers"] == [
                    {"kind": "quarantine", "operation_id": operation}
                ]
                return operation

    operation = asyncio.run(first())

    async def restart():
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine([], [Resource("arm")], journal=journal) as engine:
                run = engine.get_run_by_key("key")
                assert run.operation_id("arm/move") == operation
                assert (
                    run.explain("arm/move")["blockers"][0]["operation_id"] == operation
                )
                engine.confirm_quiescent(
                    operation, evidence="controller confirmed stop"
                )
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0

    asyncio.run(restart())


def test_recovery_checks_aggregate_leases_before_reducing_capacity(tmp_path):
    path = tmp_path / "cell.db"

    async def first():
        async def lost(context, inputs):
            raise ExecutionUncertain("remote state unavailable")

        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("lost", lost, (ResourceClaim("gpu"),))],
                [Resource("gpu", 2)],
                journal=journal,
            ) as engine:
                for node in ("a", "b"):
                    run = engine.submit(Workflow((Node(node, "lost"),)))
                    assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
                assert engine.resource_snapshot()["resources"]["gpu"]["used"] == 2

    asyncio.run(first())

    async def failed_startup():
        with SQLiteJournal(path) as journal:
            engine = WorkflowEngine([], [Resource("gpu", 1)], journal=journal)
            with pytest.raises(ValueError, match="exceed resource capacity"):
                await engine.__aenter__()
            assert engine.resource_snapshot()["recovery_pending"]
            assert not engine._leases
            assert len(journal._load()) == 2

    asyncio.run(failed_startup())

    async def valid_startup():
        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [], [Resource("gpu", 2)], journal=journal
            ) as engine:
                assert engine.resource_snapshot()["resources"]["gpu"]["used"] == 2
                for operation in engine.resource_snapshot()["quarantined"]:
                    engine.confirm_quiescent(
                        operation, evidence="GPU job confirmed stopped"
                    )
                assert journal.prune_completed_runs(0) == 2

    asyncio.run(valid_startup())


def test_hard_exit_during_schema_creation_can_initialize_cleanly_on_restart(tmp_path):
    path = tmp_path / "cell.db"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os, sqlite3, sys
from orchestrion.workflow import SQLiteJournal
original = sqlite3.connect
class Interrupted(sqlite3.Connection):
    def execute(self, statement, *args, **kwargs):
        if statement.startswith("CREATE TABLE nodes"):
            os._exit(73)
        return super().execute(statement, *args, **kwargs)
def connect(*args, **kwargs):
    return original(*args, factory=Interrupted, **kwargs)
sqlite3.connect = connect
SQLiteJournal(sys.argv[1])
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 73, result.stderr
    with SQLiteJournal(path) as journal:
        assert journal.get_run("missing") is None
        assert journal._load() == []
        assert journal._db.execute("PRAGMA user_version").fetchone()[0] == 1


@pytest.mark.parametrize(
    "version,mode,sync",
    [
        ((3, 40, 1), "delete", 3),
        ((3, 44, 5), "delete", 3),
        ((3, 44, 6), "wal", 2),
        ((3, 45, 0), "delete", 3),
        ((3, 50, 4), "delete", 3),
        ((3, 50, 7), "wal", 2),
        ((3, 51, 0), "delete", 3),
        ((3, 51, 2), "delete", 3),
        ((3, 51, 3), "wal", 2),
        ((3, 53, 1), "wal", 2),
    ],
)
def test_auto_journal_mode_keeps_durability_and_uses_only_fixed_wal_versions(
    tmp_path, version, mode, sync
):
    with patch.object(sqlite3, "sqlite_version_info", version):
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            assert journal.journal_mode == mode
            assert journal.synchronous == sync
            assert journal._db.execute("PRAGMA journal_mode").fetchone()[0] == mode
            assert journal._db.execute("PRAGMA synchronous").fetchone()[0] == sync


def test_explicit_wal_rejects_an_unfixed_sqlite_before_creating_files(tmp_path):
    path = tmp_path / "cell.db"
    with patch.object(sqlite3, "sqlite_version_info", (3, 50, 4)):
        with pytest.raises(JournalError, match="WAL-reset fix"):
            SQLiteJournal(path, journal_mode="wal")
    assert not path.exists() and not (tmp_path / "cell.db.owner").exists()


def test_result_identity_survives_switching_durable_journal_modes(tmp_path):
    path = tmp_path / "cell.db"
    workflow = Workflow((Node("n", "echo", {"value": 42}),))
    ids = []
    calls = []

    async def scenario(mode):
        async def record(context, inputs):
            calls.append(context.operation_id)
            return inputs

        with SQLiteJournal(path, journal_mode=mode) as journal:
            async with WorkflowEngine(
                [Action("echo", record)], journal=journal
            ) as engine:
                run = engine.submit(workflow, idempotency_key="key")
                ids.append(run.run_id)
                assert (await run.wait(1)).nodes["n"].output == {"value": 42}

    for mode in ("delete", "auto", "delete"):
        asyncio.run(scenario(mode))
    assert len(set(ids)) == 1
    assert len(calls) == 1


def test_memory_workflows_remain_usable_when_python_omits_sqlite_support(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio, builtins, sys
original = builtins.__import__
def without_sqlite(name, *args, **kwargs):
    if name == "sqlite3":
        raise ImportError("no _sqlite3 extension")
    return original(name, *args, **kwargs)
builtins.__import__ = without_sqlite
from orchestrion.workflow import Action, JournalError, Node, RunStatus, SQLiteJournal, Workflow, WorkflowEngine
async def echo(context, inputs):
    return inputs
async def main():
    async with WorkflowEngine([Action("echo", echo)]) as engine:
        run = engine.submit(Workflow((Node("n", "echo"),)))
        assert (await run.wait(1)).status == RunStatus.SUCCEEDED
asyncio.run(main())
try:
    SQLiteJournal(sys.argv[1])
except JournalError as error:
    assert "no SQLite support" in str(error)
else:
    raise AssertionError("journal unexpectedly opened")
""",
            str(tmp_path / "cell.db"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "cell.db").exists()


@pytest.mark.parametrize("stop", ["cancel", "failure"])
def test_graph_stop_is_visible_as_one_atomic_durable_transition(tmp_path, stop):
    async def scenario():
        async def fail(context, inputs):
            raise RuntimeError("device confirmed failure")

        path = tmp_path / "cell.db"
        workflow = Workflow(
            tuple(
                Node(str(i), "fail", after=(str(i - 1),) if i else ())
                for i in range(40)
            )
        )
        with SQLiteJournal(path, max_events_per_run=3) as journal:
            async with WorkflowEngine(
                [Action("fail", fail)], journal=journal
            ) as engine:
                original = journal._event
                observations = []

                def inspect(db, run_id, node_id, kind, detail=None, *, trim=True):
                    original(db, run_id, node_id, kind, detail, trim=trim)
                    if kind == "settled":
                        with sqlite3.connect(path) as reader:
                            observations.append(
                                reader.execute(
                                    "SELECT count(*) FROM nodes WHERE status IN "
                                    "('failed','skipped','cancelled')"
                                ).fetchone()[0]
                            )

                with patch.object(journal, "_event", side_effect=inspect):
                    run = engine.submit(workflow)
                    if stop == "cancel":
                        run.cancel()
                    result = await run.wait(2)
                assert observations == [0] * len(workflow.nodes)
                assert all(node.status.terminal for node in result.nodes.values())
                assert all(
                    NodeStatus(node["status"]).terminal
                    for node in journal.get_run(run.run_id)["nodes"]
                )
                events = journal.events(run.run_id)
                assert len(events) == 3
                assert [event["kind"] for event in events] == [
                    "settled", "settled", "finished"
                ]

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["cancel", "failure"])
def test_fault_mid_graph_stop_rolls_back_entire_durable_transition(tmp_path, stop):
    path = tmp_path / "cell.db"

    async def first():
        async def fail(context, inputs):
            raise RuntimeError("device confirmed failure")

        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("fail", fail, (ResourceClaim("arm"),))],
                [Resource("arm")],
                journal=journal,
            ) as engine:
                original = journal._event

                def disk_full(db, run_id, node_id, kind, detail=None, *, trim=True):
                    if kind == "settled" and node_id == "5":
                        raise JournalError("disk full mid graph transition")
                    original(db, run_id, node_id, kind, detail, trim=trim)

                with patch.object(journal, "_event", side_effect=disk_full):
                    run = engine.submit(
                        Workflow(
                            tuple(
                                Node(str(i), "fail", after=(str(i - 1),) if i else ())
                                for i in range(12)
                            )
                        ),
                        idempotency_key="batch",
                    )
                    if stop == "cancel":
                        run.cancel()
                    await run.wait(2)
                record = journal.get_run(run.run_id)
                assert record["finished_at"] is None
                assert record["nodes"][0]["status"] == (
                    "pending" if stop == "cancel" else "running"
                )
                assert all(node["status"] == "pending" for node in record["nodes"][1:])
                assert not any(
                    e["kind"] == "settled" for e in journal.events(run.run_id)
                )

    async def restart():
        with SQLiteJournal(path) as journal:
            async with WorkflowEngine([], [Resource("arm")], journal=journal) as engine:
                restored = engine.get_run_by_key("batch").snapshot()
                assert restored.nodes["0"].status == (
                    NodeStatus.CANCELLED if stop == "cancel" else NodeStatus.UNKNOWN
                )
                assert all(
                    restored.nodes[str(i)].status == NodeStatus.CANCELLED
                    for i in range(1, 12)
                )
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == int(
                    stop == "failure"
                )

    asyncio.run(first())
    asyncio.run(restart())


@pytest.mark.parametrize(
    "corruption,expected",
    [
        ("UPDATE nodes SET status='running', lease_active=0", "missing its lease"),
        ("UPDATE nodes SET lease_active=0", "no reconciliation evidence"),
        (
            "UPDATE nodes SET reconciled_at=1, reconciliation_evidence='idle'",
            "inconsistent operation",
        ),
        ("UPDATE runs SET status='succeeded'", "outcome disagrees"),
    ],
)
@pytest.mark.parametrize("key", [None, "key"])
def test_inconsistent_recovery_records_are_rejected_before_database_changes(
    tmp_path,
    corruption,
    expected,
    key,
):
    path = tmp_path / "cell.db"

    async def first():
        async def uncertain(context, inputs):
            raise ExecutionUncertain("device is unreachable")

        with SQLiteJournal(path) as journal:
            async with WorkflowEngine(
                [Action("uncertain", uncertain, (ResourceClaim("arm"),))],
                [Resource("arm")],
                journal=journal,
            ) as engine:
                run = engine.submit(
                    Workflow((Node("n", "uncertain"),)), idempotency_key=key
                )
                assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION

    asyncio.run(first())
    with sqlite3.connect(path) as db:
        db.execute(corruption)
        if "status='running'" in corruption:
            db.execute("UPDATE runs SET status='running', finished_at=NULL")
    with SQLiteJournal(path) as journal:
        before = journal._load()

        async def restart():
            engine = WorkflowEngine([], [Resource("arm")], journal=journal)
            with pytest.raises(JournalError, match=expected):
                await engine.__aenter__()
            assert engine._closed
            assert not engine._leases
            assert engine.resource_snapshot()["recovery_pending"]

        asyncio.run(restart())
        assert journal._load() == before
