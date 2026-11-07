"""Workflow diagnostics, event history, and bounded SDK progress delivery."""

import asyncio
import json
import threading
import time

import pytest

from orchestrion.workflow import (
    Action,
    ActionContext,
    ExecutionUncertain,
    Node,
    NodeStatus,
    OutputRef,
    Resource,
    ResourceClaim,
    RunStatus,
    Workflow,
    WorkflowEngine,
)


async def echo(context, inputs):
    return inputs


def test_explain_distinguishes_dispatch_dependencies_and_concurrency():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold(context, inputs):
            entered.set()
            await asyncio.wait_for(release.wait(), 1)
            return {"pose": [1]}

        async with WorkflowEngine(
            [Action("hold", hold), Action("echo", echo)], max_concurrency=1
        ) as engine:
            workflow = Workflow(
                (
                    Node("hold", "hold"),
                    Node("dependent", "echo", {"pose": OutputRef("hold", ("pose",))}),
                    Node("parallel", "echo"),
                )
            )
            run = engine.submit(workflow, idempotency_key="cycle")
            try:
                assert run.explain("hold")["blockers"] == [{"kind": "dispatch"}]
                assert run.explain("dependent")["blockers"] == [
                    {"kind": "dependencies", "nodes": ["hold"]}
                ]
                await asyncio.wait_for(entered.wait(), 1)
                assert run.explain("hold")["blockers"] == []
                assert run.explain("parallel")["blockers"] == [
                    {"kind": "concurrency", "running": 1, "limit": 1}
                ]
                snapshot = engine.scheduler_snapshot()
                assert snapshot["active_runs"] == 1
                assert snapshot["running_actions"] == 1
                assert snapshot["nodes"]["pending"] == 1
                assert snapshot["nodes"]["ready"] == 1
                assert snapshot["nodes"]["running"] == 1
                assert snapshot["idempotency_keys"] == {
                    "retained": 1,
                    "limit": 1024,
                    "prunable": 0,
                }
                json.dumps(snapshot, allow_nan=False)
                json.dumps(run.explain("dependent"), allow_nan=False)
                snapshot["nodes"]["running"] = 999
                assert engine.scheduler_snapshot()["nodes"]["running"] == 1
                with pytest.raises(KeyError):
                    run.explain("missing")
            finally:
                release.set()
            await run.wait(1)
            assert run.explain("dependent")["blockers"] == []
            assert engine.scheduler_snapshot()["idempotency_keys"]["prunable"] == 1
        assert engine.scheduler_snapshot()["closed"] is True
        assert engine.scheduler_snapshot()["active_runs"] == 0

    asyncio.run(scenario())


def test_explain_atomic_queue_head_even_when_requested_resource_is_free():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def hold(context, inputs):
            entered.set()
            await asyncio.wait_for(release.wait(), 1)

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [Action("hold", hold, (ResourceClaim("b"),)), Action("record", record)],
            [Resource("a"), Resource("b")],
        ) as engine:
            owner = engine.submit(Workflow((Node("owner", "hold"),)))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                first = engine.submit(
                    Workflow(
                        (
                            Node(
                                "both",
                                "record",
                                claims=(ResourceClaim("a"), ResourceClaim("b")),
                            ),
                        )
                    )
                )
                second = engine.submit(
                    Workflow((Node("single", "record", claims=(ResourceClaim("a"),)),))
                )
                before = [
                    list(entries) for entries in engine._resource_waiters.values()
                ]
                assert first.explain("both")["blockers"] == [
                    {
                        "kind": "resource_capacity",
                        "resource": "b",
                        "required": 1,
                        "available": 0,
                        "owners": [
                            {
                                "operation_id": owner.run_id + "/owner",
                                "units": 1,
                                "quarantined": False,
                            }
                        ],
                    }
                ]
                assert second.explain("single")["blockers"] == [
                    {
                        "kind": "resource_queue",
                        "resource": "a",
                        "ahead": first.run_id + "/both",
                    }
                ]
                state = engine.scheduler_snapshot()
                assert state["resources"]["a"]["used"] == 0
                assert state["resources"]["a"]["waiting"] == 2
                assert (
                    state["resources"]["a"]["next_operation"] == first.run_id + "/both"
                )
                assert state["resources"]["b"]["waiting"] == 1
                state["resources"]["b"]["owners"][0]["units"] = 100
                assert (
                    engine.scheduler_snapshot()["resources"]["b"]["owners"][0]["units"]
                    == 1
                )
                assert before == list(engine._resource_waiters.values())
            finally:
                release.set()
            await asyncio.gather(owner.wait(1), first.wait(1), second.wait(1))
            assert calls == ["both", "single"]
            assert engine.scheduler_snapshot()["resources"]["a"]["waiting"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("unrelated_resources", [0, 64])
def test_explain_preserves_each_owner_for_multiple_blocked_claims(unrelated_resources):
    async def scenario():
        names = ("arm", "tool", "camera")
        entered = {name: asyncio.Event() for name in names}
        release = asyncio.Event()

        async def hold(context, inputs):
            entered[context.node_id].set()
            await asyncio.wait_for(release.wait(), 2)

        resources = [Resource(name) for name in names] + [
            Resource("unused-" + str(index)) for index in range(unrelated_resources)
        ]
        async with WorkflowEngine(
            [Action("hold", hold), Action("echo", echo)],
            resources,
            max_concurrency=3,
        ) as engine:
            owner_nodes = tuple(
                Node(name, "hold", claims=(ResourceClaim(name),)) for name in names
            )
            owner = engine.submit(Workflow(owner_nodes))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(event.wait() for event in entered.values())), 1
                )
                blocked_node = Node(
                    "both", "echo", claims=(ResourceClaim("arm"), ResourceClaim("tool"))
                )
                waiter = engine.submit(Workflow((blocked_node,)))
                expected = waiter.explain("both")
                diagnostic = waiter.explain("both")
                capacity_blockers = [
                    blocker
                    for blocker in diagnostic["blockers"]
                    if blocker["kind"] == "resource_capacity"
                ]
                assert [blocker["resource"] for blocker in capacity_blockers] == [
                    "arm",
                    "tool",
                ]
                for blocker in capacity_blockers:
                    assert blocker["owners"] == [
                        {
                            "operation_id": owner.operation_id(blocker["resource"]),
                            "units": 1,
                            "quarantined": False,
                        }
                    ]
                snapshot = engine.scheduler_snapshot()
                assert snapshot["resources"]["camera"]["owners"][0]["operation_id"] == (
                    owner.operation_id("camera")
                )
                capacity_blockers[0]["owners"][0]["units"] = 99
                assert waiter.explain("both") == expected
            finally:
                release.set()
            await asyncio.gather(owner.wait(1), waiter.wait(1))

    asyncio.run(scenario())


def test_explain_uses_the_ready_heap_head_without_scanning_the_backlog(monkeypatch):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold(context, inputs):
            entered.set()
            await asyncio.wait_for(release.wait(), 2)

        async with WorkflowEngine(
            [Action("hold", hold, (ResourceClaim("arm"),))], [Resource("arm")]
        ) as engine:
            owner = engine.submit(Workflow((Node("owner", "hold"),)))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                run = engine.submit(Workflow(tuple(
                    Node("wait-{}".format(i), "hold") for i in range(100)
                )))
                before = list(engine._resource_waiters["arm"])
                calls = 0
                original = engine._is_ready

                def count_readiness(entry):
                    nonlocal calls
                    calls += 1
                    return original(entry)

                monkeypatch.setattr(engine, "_is_ready", count_readiness)
                explanation = run.explain("wait-99")
                assert calls == 1
                assert explanation["blockers"][-1] == {
                    "kind": "resource_queue",
                    "resource": "arm",
                    "ahead": run.operation_id("wait-0"),
                }
                assert engine._resource_waiters["arm"] == before
            finally:
                release.set()
            await asyncio.gather(owner.wait(2), run.wait(2))

    asyncio.run(scenario())


def test_explain_cancellation_keeps_stop_confirmation_and_resource_owner_visible():
    async def scenario():
        entered, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def move(context, inputs):
            entered.set()
            await context.wait_cancelled()
            stopping.set()
            await asyncio.wait_for(release.wait(), 1)
            context.raise_if_cancelled()

        async with WorkflowEngine(
            [Action("move", move, (ResourceClaim("arm"),))], [Resource("arm")]
        ) as engine:
            run = engine.submit(Workflow((Node("move", "move"),)))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                run.cancel()
                await asyncio.wait_for(stopping.wait(), 1)
                assert run.explain("move")["blockers"] == [
                    {"kind": "stop_confirmation", "reason": "cancelled"}
                ]
                snapshot = engine.scheduler_snapshot()
                assert snapshot["running_actions"] == 1
                assert snapshot["nodes"]["cancelling"] == 1
                assert (
                    snapshot["resources"]["arm"]["owners"][0]["operation_id"]
                    == run.run_id + "/move"
                )
            finally:
                release.set()
            assert (await run.wait(1)).status == RunStatus.CANCELLED
            assert run.explain("move")["blockers"] == []

    asyncio.run(scenario())


def test_completed_unknown_is_visible_until_external_reconciliation():
    async def scenario():
        async def lost(context, inputs):
            raise ExecutionUncertain("connection lost")

        engine = WorkflowEngine(
            [
                Action("lost", lost, (ResourceClaim("arm"),)),
                Action("move", echo, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
        )
        async with engine:
            lost_run = engine.submit(
                Workflow((Node("lost", "lost"),)), idempotency_key="cycle"
            )
            await lost_run.wait(1)
            operation_id = lost_run.run_id + "/lost"
            assert lost_run.explain("lost")["blockers"] == [
                {"kind": "quarantine", "operation_id": operation_id}
            ]
            snapshot = engine.scheduler_snapshot()
            assert snapshot["active_runs"] == 0
            assert snapshot["running_actions"] == 0
            assert snapshot["idempotency_keys"]["prunable"] == 0
            assert snapshot["resources"]["arm"]["owners"] == [
                {"operation_id": operation_id, "units": 1, "quarantined": True}
            ]
            next_run = engine.submit(Workflow((Node("next", "move"),)))
            blocker = next_run.explain("next")["blockers"][0]
            assert blocker["kind"] == "resource_capacity"
            assert blocker["owners"][0]["quarantined"] is True
            engine.confirm_quiescent(operation_id, evidence="controller confirmed stop")
            assert lost_run.explain("lost")["status"] == "unknown"
            assert lost_run.explain("lost")["blockers"] == []
            assert engine.scheduler_snapshot()["idempotency_keys"]["prunable"] == 1
            assert (await next_run.wait(1)).status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


def test_diagnostics_reject_access_from_a_foreign_event_loop():
    engine = WorkflowEngine([Action("echo", echo)])

    async def first_loop():
        async with engine:
            run = engine.submit(Workflow((Node("n", "echo"),)), idempotency_key="key")
            await run.wait(1)
            return run

    run = asyncio.run(first_loop())

    async def second_loop():
        for function in [
            engine.scheduler_snapshot,
            lambda: run.explain("n"),
            lambda: engine.get_run_by_key("key"),
            lambda: engine.prune_idempotency_keys(0),
        ]:
            with pytest.raises(RuntimeError, match="original event loop"):
                function()

    asyncio.run(second_loop())


@pytest.mark.parametrize("history_limit", [3, 64])
def test_sdk_burst_bounds_pending_callbacks_and_retains_latest_progress(history_limit):
    async def scenario():
        loop = asyncio.get_running_loop()
        original_notify = loop.call_soon_threadsafe
        notifications = []
        thread_errors = []
        observations = []
        report_count = 20000

        async def adapter(context, inputs):
            def notify(callback, *args, **kwargs):
                if callback == context._feedback_channel._drain:
                    notifications.append(callback)
                return original_notify(callback, *args, **kwargs)

            def produce():
                try:
                    for index in range(report_count):
                        payload = {"progress": [index]}
                        context.report(payload)
                        payload["progress"].clear()
                except BaseException as exc:
                    thread_errors.append(exc)

            loop.call_soon_threadsafe = notify
            thread = threading.Thread(target=produce)
            try:
                thread.start()
                # Deliberately stall this loop while the bounded producer bursts.
                thread.join(2)
                assert not thread.is_alive()
                assert not thread_errors
                observations.append(run.explain("sdk")["feedback"])
            finally:
                loop.call_soon_threadsafe = original_notify
                thread.join(2)
            # Completing before the drain callback runs must flush accepted data.

        async with WorkflowEngine(
            [Action("sdk", adapter)],
            max_pending_feedback_per_action=8,
            max_events_per_run=history_limit,
        ) as engine:
            run = engine.submit(Workflow((Node("sdk", "sdk"),)))
            result = await run.wait(3)
            assert result.status == RunStatus.SUCCEEDED
            assert len(notifications) == 1
            assert observations == [{"pending": 8, "dropped": report_count - 8}]
            assert result.nodes["sdk"].feedback_dropped == report_count - 8
            assert run.explain("sdk")["feedback"] == {
                "pending": 0,
                "dropped": report_count - 8,
            }
            feedback = [
                event.detail for event in run.events() if event.kind == "feedback"
            ]
            assert feedback[-1] == {"progress": [report_count - 1]}
            if history_limit == 64:
                assert feedback == [
                    {"progress": [index]}
                    for index in range(report_count - 8, report_count)
                ]
                assert [
                    event.detail
                    for event in run.events()
                    if event.kind == "feedback_overflow"
                ] == [{"dropped": report_count - 8}]
            events = run.events()
            await asyncio.sleep(0)
            assert run.events() == events

    asyncio.run(scenario())


def test_multiple_sdk_threads_share_one_bounded_queue_without_lost_wakeups():
    async def scenario():
        thread_errors = []

        async def adapter(context, inputs):
            def produce(producer):
                try:
                    for index in range(100):
                        context.report({"producer": producer, "index": index})
                except BaseException as exc:
                    thread_errors.append(exc)

            threads = [
                threading.Thread(target=produce, args=(producer,))
                for producer in range(4)
            ]
            try:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(1)
                    assert not thread.is_alive()
                assert not thread_errors
                await asyncio.sleep(0)
            finally:
                for thread in threads:
                    thread.join(1)

        async with WorkflowEngine(
            [Action("sdk", adapter)],
            max_pending_feedback_per_action=512,
        ) as engine:
            run = engine.submit(Workflow((Node("sdk", "sdk"),)))
            result = await run.wait(2)
            assert result.status == RunStatus.SUCCEEDED
            assert result.nodes["sdk"].feedback_dropped == 0
            feedback = [
                event.detail for event in run.events() if event.kind == "feedback"
            ]
            assert len(feedback) == 400
            for producer in range(4):
                assert [
                    item["index"] for item in feedback if item["producer"] == producer
                ] == list(range(100))

    asyncio.run(scenario())


def test_continuous_sdk_feedback_accounts_for_every_delivered_or_dropped_sample():
    async def scenario():
        loop = asyncio.get_running_loop()
        finished = asyncio.Event()
        errors = []
        count = 10000

        async def adapter(context, inputs):
            def produce():
                try:
                    for index in range(count):
                        context.report(index)
                        if index % 100 == 0:
                            time.sleep(0)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    loop.call_soon_threadsafe(finished.set)

            thread = threading.Thread(target=produce)
            thread.start()
            try:
                await asyncio.wait_for(finished.wait(), 2)
            finally:
                thread.join(1)
                assert not thread.is_alive()

        async with WorkflowEngine(
            [Action("sdk", adapter)],
            max_pending_feedback_per_action=8,
            max_events_per_run=count + 10,
        ) as engine:
            run = engine.submit(Workflow((Node("sdk", "sdk"),)))
            result = await run.wait(3)
            assert result.status == RunStatus.SUCCEEDED
            assert not errors
            samples = [
                event.detail for event in run.events() if event.kind == "feedback"
            ]
            dropped = sum(
                event.detail["dropped"]
                for event in run.events()
                if event.kind == "feedback_overflow"
            )
            assert len(samples) + dropped == count
            assert result.nodes["sdk"].feedback_dropped == dropped
            assert samples == sorted(set(samples))
            assert samples[-1] == count - 1

    asyncio.run(scenario())


@pytest.mark.parametrize("initial_count", [1, 32, 100])
def test_feedback_drains_in_batches_and_wakes_again_after_becoming_empty(initial_count):
    async def scenario():
        delivered = []
        turns = []
        errors = []
        context = ActionContext("run", "node", delivered.append)

        def produce(start, count):
            try:
                for index in range(start, start + count):
                    context.report(index)
            except BaseException as exc:
                errors.append(exc)

        try:
            thread = threading.Thread(target=produce, args=(0, initial_count))
            thread.start()
            thread.join(1)
            assert not thread.is_alive()
            asyncio.get_running_loop().call_soon(lambda: turns.append(len(delivered)))
            for _ in range(5):
                await asyncio.sleep(0)
            assert delivered == list(range(initial_count))
            assert turns == [min(32, initial_count)]
            thread = threading.Thread(target=produce, args=(initial_count, 2))
            thread.start()
            thread.join(1)
            assert not thread.is_alive()
            await asyncio.sleep(0)
            assert delivered == list(range(initial_count + 2))
            assert not errors
        finally:
            context._finish()

    asyncio.run(scenario())


def test_late_sdk_feedback_after_loop_shutdown_is_ignored():
    async def create_context():
        return ActionContext("run", "node", lambda value: None)

    context = asyncio.run(create_context())
    errors = []

    def late_callback():
        try:
            context.report({"late": True})
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=late_callback)
    thread.start()
    thread.join(1)
    assert not thread.is_alive()
    assert not errors
    assert context._feedback_snapshot()["pending"] == 0


def test_late_feedback_during_snapshot_is_discarded_after_completion():
    async def scenario():
        copying, release = threading.Event(), threading.Event()
        delivered, errors = [], []
        context = ActionContext("run", "node", delivered.append)

        class DelayedMapping(dict):
            def items(self):
                copying.set()
                assert release.wait(1)
                return super().items()

        def callback():
            try:
                context.report(DelayedMapping(value=1))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=callback)
        thread.start()
        try:
            assert copying.wait(1)
            context._finish()
        finally:
            release.set()
            thread.join(1)
        assert not thread.is_alive()
        assert not errors
        await asyncio.sleep(0)
        assert delivered == []

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0, -1, True, None, 1.5])
def test_invalid_feedback_capacity_is_rejected(value):
    with pytest.raises(ValueError, match="max_pending_feedback_per_action"):
        WorkflowEngine([], max_pending_feedback_per_action=value)

    async def scenario():
        with pytest.raises(ValueError, match="max_pending_feedback"):
            ActionContext("run", "node", lambda value: None, max_pending_feedback=value)

    asyncio.run(scenario())


def test_late_feedback_does_not_reopen_or_extend_terminal_history():
    async def scenario():
        async def report(context, inputs):
            context.report({"value": "active"})
            asyncio.get_running_loop().call_soon(context.report, {"value": "late"})

        async with WorkflowEngine([Action("report", report)]) as engine:
            run = engine.submit(Workflow((Node("report", "report"),)))
            await run.wait(1)
            feedback = [
                event.detail for event in run.events() if event.kind == "feedback"
            ]
            assert feedback == [{"value": "active"}]

    asyncio.run(scenario())


def test_sdk_thread_feedback_is_snapshotted_and_delivered_on_the_event_loop():
    async def scenario():
        loop = asyncio.get_running_loop()
        delivered = asyncio.Event()
        callback_thread = None

        async def adapter(context, inputs):
            nonlocal callback_thread

            def sdk_callback():
                for index in range(10):
                    payload = {"progress": [index]}
                    context.report(payload)
                    payload["progress"].clear()
                loop.call_soon_threadsafe(delivered.set)

            callback_thread = threading.Thread(target=sdk_callback)
            callback_thread.start()
            await asyncio.wait_for(delivered.wait(), 1)

        async with WorkflowEngine([Action("adapter", adapter)]) as engine:
            try:
                run = engine.submit(Workflow((Node("adapter", "adapter"),)))
                assert (await run.wait(1)).status == RunStatus.SUCCEEDED
                feedback = [
                    event.detail for event in run.events() if event.kind == "feedback"
                ]
                assert feedback == [{"progress": [i]} for i in range(10)]
            finally:
                if callback_thread is not None:
                    callback_thread.join(1)
                    assert not callback_thread.is_alive()

    asyncio.run(scenario())


@pytest.mark.parametrize("followup_count", [8, 64])
def test_incremental_events_preserve_order_and_isolation_across_history_eviction(
    followup_count,
):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        initial_count, history_limit = 16, 32

        async def report(context, inputs):
            for step in range(initial_count):
                context.report({"progress": [step]})
            entered.set()
            await asyncio.wait_for(release.wait(), 2)
            for step in range(initial_count, initial_count + followup_count):
                context.report({"progress": [step]})

        async with WorkflowEngine(
            [Action("report", report)], max_events_per_run=history_limit
        ) as engine:
            run = engine.submit(Workflow((Node("report", "report"),)))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                cursor = run.events()[-1].sequence
                tail = run.events(after=cursor - 2)
                assert [event.detail["progress"] for event in tail] == [[14], [15]]
                tail[0].detail["progress"].clear()
                assert run.events(after=cursor - 2)[0].detail["progress"] == [14]
            finally:
                release.set()
            await run.wait(1)
            recent = run.events(after=cursor)
            total_count = initial_count + followup_count
            first_step = max(initial_count, total_count - history_limit + 1)
            assert [
                event.detail["progress"][0]
                for event in recent
                if event.kind == "feedback"
            ] == list(range(first_step, total_count))
            assert len(recent) == min(followup_count + 1, history_limit)
            assert recent[-1].status == NodeStatus.SUCCEEDED
            assert (recent[0].sequence > cursor + 1) == (
                followup_count >= history_limit
            )
            assert all(
                right.sequence == left.sequence + 1
                for left, right in zip(recent, recent[1:])
            )

    asyncio.run(scenario())


def test_events_are_bounded_and_defensive_with_monotonic_sequences():
    async def scenario():
        async def report(context, inputs):
            for step in range(10):
                payload = {"progress": [step]}
                context.report(payload)
                payload["progress"].clear()
            checkpoint = {"progress": [10]}
            context.checkpoint(checkpoint)
            checkpoint["progress"].clear()

        async with WorkflowEngine(
            [Action("report", report)], max_events_per_run=4
        ) as engine:
            run = engine.submit(Workflow((Node("report", "report"),)))
            result = await run.wait(1)
            events = run.events()
            assert len(events) == 4
            assert events[0].sequence > 1
            assert events[-1].status == NodeStatus.SUCCEEDED
            assert run.events(after=events[-2].sequence) == events[-1:]
            assert run.events(after=events[-1].sequence) == ()
            assert run.events(after=events[-1].sequence + 100) == ()
            events[0].detail["progress"].clear()
            assert run.events()[0].detail["progress"]
            checkpoint_event = next(
                event for event in events if event.kind == "checkpoint"
            )
            assert checkpoint_event.detail == {"progress": [10]}
            checkpoint_event.detail["progress"].clear()
            assert run.snapshot().nodes["report"].checkpoint == {"progress": [10]}
            result.nodes["report"].checkpoint["progress"].append(11)
            assert next(
                event.detail for event in run.events() if event.kind == "checkpoint"
            ) == {"progress": [10]}

    asyncio.run(scenario())
