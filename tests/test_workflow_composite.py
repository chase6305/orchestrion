"""Continuous resource ownership across the steps of a composite capability."""

import asyncio
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest

from orchestrion.workflow import (
    Action,
    ActionContext,
    ActionOutcome,
    BlockingAction,
    ExecutionUncertain,
    JournalInUse,
    Node,
    NodeStatus,
    OutputRef,
    PayloadLimitError,
    Resource,
    ResourceClaim,
    RunStatus,
    SQLiteJournal,
    Workflow,
    WorkflowEngine,
    workflow_action,
)


def test_scope_keeps_resources_between_steps_and_allows_unrelated_work():
    async def scenario():
        gap = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def pick(context, inputs):
            calls.append((context.operation_id, "pick"))
            return {"object": "part"}

        async def inspect(context, inputs):
            gap.set()
            await asyncio.wait_for(release.wait(), 1)
            context.report({"checked": inputs["object"]})
            return inputs

        async def place(context, inputs):
            calls.append((context.operation_id, "place"))
            return inputs

        async def other(context, inputs):
            calls.append((context.operation_id, "other"))

        composite = workflow_action(
            "transfer",
            Workflow(
                (
                    Node("pick", "pick"),
                    Node(
                        "inspect", "inspect", {"object": OutputRef("pick", ("object",))}
                    ),
                    Node(
                        "place", "place", {"object": OutputRef("inspect", ("object",))}
                    ),
                )
            ),
            [
                Action("pick", pick, (ResourceClaim("arm"),)),
                Action("inspect", inspect),
                Action("place", place, (ResourceClaim("arm"),)),
            ],
            [Resource("arm"), Resource("zone")],
        )
        async with WorkflowEngine(
            [
                composite,
                Action("other", other, (ResourceClaim("zone"),)),
                Action("unrelated", other),
            ],
            [Resource("arm"), Resource("zone")],
        ) as engine:
            transfer = engine.submit(Workflow((Node("transfer", "transfer"),)))
            try:
                await asyncio.wait_for(gap.wait(), 1)
                intruder = engine.submit(Workflow((Node("intruder", "other"),)))
                unrelated = engine.submit(Workflow((Node("unrelated", "unrelated"),)))
                assert (await unrelated.wait(1)).status == RunStatus.SUCCEEDED
                assert not intruder.done
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
                assert engine.resource_snapshot()["resources"]["zone"]["used"] == 1
                assert (
                    intruder.explain("intruder")["blockers"][0]["kind"]
                    == "resource_capacity"
                )
            finally:
                release.set()
            result = await transfer.wait(1)
            await intruder.wait(1)
            assert result.status == RunStatus.SUCCEEDED
            assert result.nodes["transfer"].output["nodes"]["place"]["output"] == {
                "object": "part"
            }
            assert [kind for _, kind in calls] == ["pick", "other", "place", "other"]
            assert calls[0][0] == transfer.run_id + "/transfer/pick"
            assert calls[2][0] == transfer.run_id + "/transfer/place"
            feedback = [
                event.detail for event in transfer.events() if event.kind == "feedback"
            ]
            assert {
                "scope_node": "inspect",
                "feedback": {"checked": "part"},
            } in feedback

    asyncio.run(scenario())


def test_parallel_inner_nodes_share_reserved_capacity_without_reacquiring_outer_locks():
    async def scenario():
        both = asyncio.Event()
        release = asyncio.Event()
        running = 0

        async def work(context, inputs):
            nonlocal running
            running += 1
            if running == 2:
                both.set()
            await asyncio.wait_for(release.wait(), 1)
            running -= 1

        scoped = workflow_action(
            "batch",
            Workflow((Node("a", "work"), Node("b", "work"))),
            [Action("work", work, (ResourceClaim("gpu"),))],
            [Resource("gpu", 2)],
        )
        async with WorkflowEngine([scoped], [Resource("gpu", 2)]) as engine:
            run = engine.submit(Workflow((Node("batch", "batch"),)))
            try:
                await asyncio.wait_for(both.wait(), 1)
                assert engine.resource_snapshot()["resources"]["gpu"]["used"] == 2
            finally:
                release.set()
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


@pytest.mark.parametrize("uncertain", [False, True])
def test_scope_cancellation_retains_all_resources_until_inner_cleanup(uncertain):
    async def scenario():
        entered, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        async def hold(context, inputs):
            entered.set()
            await context.wait_cancelled()
            stopping.set()
            await asyncio.wait_for(release.wait(), 1)
            if uncertain:
                raise ExecutionUncertain("controller did not confirm stop")
            context.raise_if_cancelled()

        async def next_step(context, inputs):
            calls.append(context.node_id)

        scoped = workflow_action(
            "transfer",
            Workflow((Node("hold", "hold"), Node("next", "next", after=("hold",)))),
            [Action("hold", hold), Action("next", next_step)],
            [Resource("arm"), Resource("zone")],
        )
        async with WorkflowEngine(
            [scoped], [Resource("arm"), Resource("zone")]
        ) as engine:
            run = engine.submit(Workflow((Node("transfer", "transfer"),)))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                run.cancel()
                await asyncio.wait_for(stopping.wait(), 1)
                assert not run.done
                assert engine.resource_snapshot()["resources"]["zone"]["used"] == 1
            finally:
                release.set()
            result = await run.wait(1)
            assert not calls
            if uncertain:
                assert result.status == RunStatus.NEEDS_ATTENTION
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
                engine.confirm_quiescent(
                    run.run_id + "/transfer", evidence="all scoped devices stopped"
                )
            else:
                assert result.status == RunStatus.CANCELLED
            assert all(
                resource["used"] == 0
                for resource in engine.resource_snapshot()["resources"].values()
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("durable", [False, True])
def test_repeated_scope_task_cancellation_keeps_inner_cleanup_in_accounting(
    depth, durable, tmp_path
):
    async def scenario():
        entered, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def device(context, inputs):
            entered.set()
            await context.wait_cancelled()
            stopping.set()
            await release.wait()
            return ActionOutcome.cancelled(context.cancel_reason)

        scoped = Action("device", device, (ResourceClaim("arm"),))
        for _ in range(depth):
            scoped = workflow_action(
                "scope",
                Workflow((Node("device", scoped.name),)),
                [scoped],
                [Resource("arm")],
            )
        storage = SQLiteJournal(tmp_path / "scope.db") if durable else nullcontext(None)
        with storage as journal:
            async with WorkflowEngine(
                [scoped], [Resource("arm")], stop_timeout=0.01, journal=journal
            ) as engine:
                run = engine.submit(Workflow((Node("scope", "scope"),)))
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                    worker = next(iter(engine._workers))
                    worker.cancel()
                    await asyncio.wait_for(stopping.wait(), 1)
                    assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
                    if journal is not None:
                        with pytest.raises(asyncio.TimeoutError):
                            await engine.aclose(0.001)
                    for _ in range(3):
                        worker.cancel()
                        await asyncio.sleep(0)
                        assert not worker.done()
                        snapshot = engine.scheduler_snapshot()
                        assert snapshot["draining_adapters"] == 1
                        assert snapshot["running_actions"] == 1
                        assert snapshot["resources"]["arm"]["used"] == 1
                        with pytest.raises(RuntimeError, match="still draining"):
                            engine.confirm_quiescent(
                                run.operation_id("scope"),
                                evidence="unconfirmed cleanup",
                            )
                        if journal is not None:
                            with pytest.raises(JournalInUse, match="drain"):
                                journal.close()
                            record = journal.get_run(run.run_id)["nodes"][0]
                            assert record["status"] == "unknown"
                            assert record["lease_active"]
                finally:
                    release.set()
                await asyncio.wait_for(asyncio.shield(worker), 1)
                # The done callback runs after the task's last step.
                await asyncio.sleep(0)
                assert engine.scheduler_snapshot()["draining_adapters"] == 0
                assert engine.scheduler_snapshot()["running_actions"] == 0
                assert run.snapshot().nodes["scope"].status == NodeStatus.UNKNOWN
                engine.confirm_quiescent(
                    run.operation_id("scope"), evidence="all scoped handlers exited"
                )
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
                if journal is not None:
                    assert not journal.get_run(run.run_id)["nodes"][0]["lease_active"]

    asyncio.run(scenario())


def test_scope_failure_stops_siblings_and_exposes_diagnostics():
    async def scenario():
        async def bad(context, inputs):
            raise ValueError("fixture rejected")

        scoped = workflow_action(
            "bad",
            Workflow((Node("bad", "bad"), Node("after", "bad", after=("bad",)))),
            [Action("bad", bad)],
            [Resource("fixture")],
        )
        async with WorkflowEngine([scoped], [Resource("fixture")]) as engine:
            run = engine.submit(Workflow((Node("bad", "bad"),)))
            result = await run.wait(1)
            assert result.status == RunStatus.FAILED
            assert "WorkflowActionError" in result.nodes["bad"].error
            assert "fixture rejected" in result.nodes["bad"].error
            assert engine.resource_snapshot()["resources"]["fixture"]["used"] == 0
            reports = [
                event.detail for event in run.events() if event.kind == "feedback"
            ]
            assert reports[-1]["scope_result"]["nodes"]["after"]["status"] == "skipped"

    asyncio.run(scenario())


@pytest.mark.parametrize("threaded", [False, True])
def test_completed_inner_context_does_not_mirror_late_feedback_into_an_active_scope(
    threaded,
):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        contexts = []

        async def first(context, inputs):
            contexts.append(context)
            context.report({"accepted": True})

        async def hold(context, inputs):
            entered.set()
            await release.wait()

        scoped = workflow_action(
            "scope",
            Workflow((Node("first", "first"), Node("hold", "hold", after=("first",)))),
            [Action("first", first), Action("hold", hold)],
        )
        async with WorkflowEngine([scoped]) as engine:
            run = engine.submit(Workflow((Node("scope", "scope"),)))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                if threaded:
                    producer = threading.Thread(
                        target=lambda: contexts[0].report({"late": True})
                    )
                    producer.start()
                    producer.join(1)
                    assert not producer.is_alive()
                    await asyncio.sleep(0)
                else:
                    contexts[0].report({"late": True})
                feedback = [
                    event.detail for event in run.events() if event.kind == "feedback"
                ]
                assert {
                    "scope_node": "first",
                    "feedback": {"accepted": True},
                } in feedback
                assert {
                    "scope_node": "first",
                    "feedback": {"late": True},
                } not in feedback
            finally:
                release.set()
            assert (await run.wait(1)).status is RunStatus.SUCCEEDED

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True])
def test_rejected_scope_checkpoint_does_not_poison_later_accepted_updates(
    durable, tmp_path
):
    async def scenario():
        rejected = []

        async def update(context, inputs):
            value = {"a": "x" * (28 * 1024), "b": "y" * (40 * 1024), "c": "ok"}[
                context.node_id
            ]
            try:
                context.checkpoint({"value": value})
            except PayloadLimitError:
                rejected.append(context.node_id)
            return {"updated": context.node_id}

        scoped = workflow_action(
            "scope",
            Workflow(
                (
                    Node("a", "update"),
                    Node("b", "update", after=("a",)),
                    Node("c", "update", after=("b",)),
                )
            ),
            [Action("update", update)],
        )
        storage = SQLiteJournal(tmp_path / "scope.db") if durable else nullcontext(None)
        with storage as journal:
            async with WorkflowEngine([scoped], journal=journal) as engine:
                run = engine.submit(Workflow((Node("scope", "scope"),)))
                result = await run.wait(2)
                assert result.status is RunStatus.SUCCEEDED
                assert rejected == ["b"]
                checkpoint = result.nodes["scope"].checkpoint
                assert set(checkpoint["scope_nodes"]) == {"a", "c"}
                assert checkpoint["scope_nodes"]["c"] == {"value": "ok"}
                if journal is not None:
                    assert (
                        journal.get_run(run.run_id)["nodes"][0]["checkpoint"]
                        == checkpoint
                    )

    asyncio.run(scenario())


def test_inner_completion_during_sdk_snapshot_suppresses_the_late_outer_copy():
    async def scenario():
        copying, copied = threading.Event(), threading.Event()
        entered, release = asyncio.Event(), asyncio.Event()
        producers = []

        class DelayedMapping(dict):
            def items(self):
                copying.set()
                assert copied.wait(2)
                return super().items()

        async def first(context, inputs):
            producer = threading.Thread(
                target=lambda: context.report(DelayedMapping(late=True))
            )
            producers.append(producer)
            producer.start()
            assert copying.wait(1)

        async def hold(context, inputs):
            entered.set()
            await release.wait()

        scoped = workflow_action(
            "scope",
            Workflow((Node("first", "first"), Node("hold", "hold", after=("first",)))),
            [Action("first", first), Action("hold", hold)],
        )
        async with WorkflowEngine([scoped]) as engine:
            run = engine.submit(Workflow((Node("scope", "scope"),)))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                copied.set()
                producers[0].join(1)
                assert not producers[0].is_alive()
                await asyncio.sleep(0)
                assert not any(
                    event.kind == "feedback"
                    and event.detail.get("feedback") == {"late": True}
                    for event in run.events()
                )
            finally:
                copied.set()
                release.set()
                for producer in producers:
                    producer.join(1)
            assert (await run.wait(1)).status is RunStatus.SUCCEEDED

    asyncio.run(scenario())


def test_completed_scope_releases_checkpoint_data_when_sdk_retains_context():
    async def scenario():
        released = threading.Event()
        contexts = []

        class Marker(str):
            def __del__(self):
                released.set()

        async def update(context, inputs):
            contexts.append(context)
            context.checkpoint({"marker": Marker("completed scope checkpoint")})

        scoped = workflow_action(
            "scope", Workflow((Node("update", "update"),)), [Action("update", update)]
        )
        engine = WorkflowEngine([scoped])
        async with engine:
            run = engine.submit(Workflow((Node("scope", "scope"),)))
            assert (await run.wait(1)).status is RunStatus.SUCCEEDED
        del run, engine
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        gc.collect()
        assert contexts
        assert released.is_set()
        with pytest.raises(RuntimeError, match="already completed"):
            contexts[0].checkpoint({"late": True})

    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True])
def test_scoped_checkpoint_preserves_the_first_captured_sdk_ticket(durable, tmp_path):
    async def scenario():
        captured = []

        class Ticket(dict):
            def items(self):
                value = "accepted-ticket" if not captured else "changed-ticket"
                captured.append(value)
                return {"ticket": value}.items()

        async def submit(context, inputs):
            assert context.checkpoint(Ticket()) is None

        scoped = workflow_action(
            "scope", Workflow((Node("submit", "submit"),)), [Action("submit", submit)]
        )
        storage = SQLiteJournal(tmp_path / "scope.db") if durable else nullcontext(None)
        with storage as journal:
            async with WorkflowEngine([scoped], journal=journal) as engine:
                run = engine.submit(Workflow((Node("scope", "scope"),)))
                result = await run.wait(1)
                assert result.status is RunStatus.SUCCEEDED
                assert captured == ["accepted-ticket"]
                expected = {"scope_nodes": {"submit": {"ticket": "accepted-ticket"}}}
                assert result.nodes["scope"].checkpoint == expected
                if journal is not None:
                    assert journal.get_run(run.run_id)["nodes"][0]["checkpoint"] == expected

    asyncio.run(scenario())


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("depth", [1, 3])
def test_scoped_feedback_reuses_the_successfully_captured_sdk_snapshot(threaded, depth):
    async def scenario():
        captures = []

        class Feedback(dict):
            def items(self):
                captures.append(len(captures) + 1)
                return {"sdk_capture": captures[-1]}.items()

        async def device(context, inputs):
            if threaded:
                producer = threading.Thread(target=lambda: context.report(Feedback()))
                producer.start()
                producer.join(1)
                assert not producer.is_alive()
            else:
                assert context.report(Feedback()) is None

        action = Action("device", device)
        for index in range(depth):
            action = workflow_action(
                "scope" + str(index), Workflow((Node("step", action.name),)), [action]
            )
        async with WorkflowEngine([action]) as engine:
            run = engine.submit(Workflow((Node("scope", action.name),)))
            assert (await run.wait(1)).status is RunStatus.SUCCEEDED
            assert captures == [1]
            received = []
            for event in run.events():
                if event.kind != "feedback":
                    continue
                detail = event.detail
                while isinstance(detail, dict) and "feedback" in detail:
                    detail = detail["feedback"]
                if isinstance(detail, dict) and "sdk_capture" in detail:
                    received.append(detail)
            assert received == [{"sdk_capture": 1}]

    asyncio.run(scenario())


def test_blocking_sdk_in_scope_uses_thread_cancellation_and_stable_operation_ids():
    async def scenario():
        release = threading.Event()
        entered = threading.Event()

        def sdk(inputs):
            entered.set()
            assert release.wait(2)
            return {"done": True}

        with ThreadPoolExecutor(1) as executor:
            scoped = workflow_action(
                "sdk",
                Workflow((Node("sdk", "sdk"),)),
                [Action("sdk", BlockingAction(sdk, executor))],
                [Resource("arm")],
            )
            async with WorkflowEngine([scoped], [Resource("arm")]) as engine:
                run = engine.submit(Workflow((Node("sdk", "sdk"),)))
                try:
                    for _ in range(100):
                        if entered.is_set():
                            break
                        await asyncio.sleep(0.001)
                    assert entered.is_set()
                    run.cancel()
                    with pytest.raises(asyncio.TimeoutError):
                        await run.wait(0.01)
                    assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
                finally:
                    release.set()
                assert (await run.wait(1)).status == RunStatus.CANCELLED

    asyncio.run(scenario())


def test_scope_definition_is_snapshotted_and_invalid_inputs_fail_before_dispatch():
    calls = []

    async def record(context, inputs):
        calls.append(inputs)
        return inputs

    source = Node("n", "record", {"value": [42]})
    scoped = workflow_action("scope", Workflow((source,)), [Action("record", record)])
    source.inputs["value"].clear()

    async def scenario():
        async with WorkflowEngine([scoped]) as engine:
            with pytest.raises(ValueError):
                engine.submit(Workflow((Node("bad", "scope", {"unexpected": 1}),)))
            assert calls == []
            run = engine.submit(Workflow((Node("scope", "scope"),)))
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED
            assert calls == [{"value": [42]}]

    asyncio.run(scenario())
    with pytest.raises(ValueError, match="unknown resource"):
        workflow_action(
            "bad",
            Workflow((Node("n", "record"),)),
            [Action("record", record, (ResourceClaim("missing"),))],
        )


def test_scope_unknown_is_durable_as_one_complete_resource_reservation(tmp_path):
    async def first():
        async def lost(context, inputs):
            raise ExecutionUncertain("lost")

        scoped = workflow_action(
            "transfer",
            Workflow((Node("move", "lost"),)),
            [Action("lost", lost)],
            [Resource("arm"), Resource("zone")],
        )
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [scoped], [Resource("arm"), Resource("zone")], journal=journal
            ) as engine:
                run = engine.submit(
                    Workflow((Node("transfer", "transfer"),)), idempotency_key="key"
                )
                assert (await run.wait(1)).nodes[
                    "transfer"
                ].status == NodeStatus.UNKNOWN
                return run.run_id

    run_id = asyncio.run(first())

    async def restart():
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [], [Resource("arm"), Resource("zone")], journal=journal
            ) as engine:
                quarantine = engine.resource_snapshot()["quarantined"]
                assert quarantine[run_id + "/transfer"]["claims"] == {
                    "arm": 1,
                    "zone": 1,
                }
                engine.confirm_quiescent(
                    run_id + "/transfer", evidence="whole scope inspected"
                )

    asyncio.run(restart())


def test_small_feedback_budget_cannot_turn_uncertain_scope_into_released_failure():
    async def scenario():
        async def lost(context, inputs):
            raise ExecutionUncertain("unknown physical state")

        scoped = workflow_action(
            "scope",
            Workflow((Node("n", "lost"),)),
            [Action("lost", lost)],
            [Resource("arm")],
        )
        async with WorkflowEngine(
            [scoped], [Resource("arm")], max_feedback_bytes_per_report=48
        ) as engine:
            run = engine.submit(Workflow((Node("scope", "scope"),)))
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("depth", [1, 3])
def test_scope_cancellation_reaches_nested_devices_before_returning(depth):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        contexts, tokens = [], []

        async def device(context, inputs):
            contexts.append(context)
            tokens.append(context._thread_cancellation())
            entered.set()
            await context.wait_cancelled()
            await release.wait()
            return ActionOutcome.cancelled(context.cancel_reason)

        scoped = Action("device", device, (ResourceClaim("arm"),))
        for _ in range(depth):
            scoped = workflow_action(
                "scope", Workflow((Node("device", scoped.name),)),
                [scoped], [Resource("arm")],
            )
        async with WorkflowEngine([scoped], [Resource("arm")]) as engine:
            try:
                run = engine.submit(Workflow((Node("scope", "scope"),)))
                await asyncio.wait_for(entered.wait(), 1)
                run.cancel()
                assert contexts[0].cancel_requested
                assert tokens[0].is_set()
                assert contexts[0].cancel_reason == "cancelled"
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            finally:
                release.set()
            assert (await run.wait(1)).status == RunStatus.CANCELLED
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0

    asyncio.run(scenario())


def test_scoped_context_propagates_internal_cancellation_to_its_delegate():
    async def scenario():
        observed = []

        async def device(context, inputs):
            token = context._thread_cancellation()
            assert context._request_cancel("adapter_interrupted")
            observed.append((context.cancel_requested, context.cancel_reason, token.is_set()))
            return ActionOutcome.cancelled(context.cancel_reason)

        scoped = workflow_action(
            "scope", Workflow((Node("device", "device"),)), [Action("device", device)]
        )
        async with WorkflowEngine([scoped]) as engine:
            run = engine.submit(Workflow((Node("scope", "scope"),)))
            assert (await run.wait(1)).status == RunStatus.CANCELLED
            assert observed == [(True, "adapter_interrupted", True)]

    asyncio.run(scenario())


def test_deep_scope_cancellation_callbacks_do_not_require_recursive_stack_growth():
    async def scenario():
        contexts = [ActionContext("run", str(i), lambda value: None) for i in range(1500)]
        try:
            for parent, child in zip(contexts, contexts[1:]):
                parent._add_cancel_callback(lambda c=child: c._request_cancel("cancelled"))
            contexts[0]._request_cancel("cancelled")
            assert all(context.cancel_requested for context in contexts)
        finally:
            for context in contexts:
                context._finish()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_failed_cancellation_callback_does_not_skip_remaining_scope_notifications(
    failure, caplog
):
    async def scenario():
        parent = ActionContext("run", "parent", lambda feedback: None)
        child = ActionContext("run", "child", lambda feedback: None)

        def broken():
            raise failure("scope cancellation callback interrupted")

        try:
            parent._add_cancel_callback(broken)
            parent._add_cancel_callback(lambda: child._request_cancel("parent stopped"))
            assert parent._request_cancel("cancelled")
            assert child.cancel_requested and child.cancel_reason == "parent stopped"
            assert "scope cancellation callback interrupted" in caplog.text
            assert not parent._request_cancel("cancelled again")
        finally:
            parent._finish()
            child._finish()

    asyncio.run(scenario())


def test_scope_fingerprint_tracks_inner_fixed_inputs_and_action_versions(tmp_path):
    async def record(context, inputs):
        return inputs

    def build(target, version="v1"):
        return workflow_action(
            "scope",
            Workflow((Node("n", "record", {"target": target}),)),
            [Action("record", record, version=version)],
        )

    assert build(1).version == build(1).version
    assert build(1).version != build(2).version
    assert build(1).version != build(1, "v2").version
    workflow = Workflow((Node("scope", "scope"),))

    async def first():
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine([build(1)], journal=journal) as engine:
                await engine.submit(workflow, idempotency_key="key").wait(1)

    asyncio.run(first())

    async def restart():
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine([build(2)], journal=journal) as engine:
                with pytest.raises(ValueError, match="different workflow"):
                    engine.submit(workflow, idempotency_key="key")
                assert engine.get_run_by_key("key").snapshot().nodes["scope"].output[
                    "nodes"
                ]["n"]["output"] == {"target": 1}

    asyncio.run(restart())


def test_scope_enforces_its_own_internal_payload_limits():
    async def record(context, inputs):
        return inputs

    with pytest.raises(ValueError, match="exceeds"):
        workflow_action(
            "scope",
            Workflow((Node("n", "record", {"image": "x" * 1000}),)),
            [Action("record", record)],
            max_input_bytes_per_node=32,
        )


@pytest.mark.parametrize("operation", ["wait", "close"])
@pytest.mark.parametrize("depth", [1, 2])
def test_scoped_actions_cannot_wait_on_or_close_an_enclosing_engine(operation, depth):
    async def scenario():
        async def recursive(context, inputs):
            with pytest.raises(RuntimeError, match="own WorkflowEngine"):
                if operation == "wait":
                    await run.wait(0.02)
                else:
                    await engine.aclose(0.02)
            return {"guarded": True}

        scoped = Action("leaf", recursive)
        for index in range(depth):
            scoped = workflow_action(
                "scope" + str(index), Workflow((Node("inner", scoped.name),)), [scoped]
            )
        async with WorkflowEngine([scoped]) as engine:
            run = engine.submit(Workflow((Node("scope", scoped.name),)))
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


def test_scoped_operation_ids_cannot_alias_literal_slashes_or_percent_encoding():
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.operation_id)
            return inputs

        scope = workflow_action(
            "scope", Workflow((Node("leaf", "record"),)), [Action("record", record)]
        )
        async with WorkflowEngine([scope, Action("record", record)]) as engine:
            run = engine.submit(
                Workflow(
                    (
                        Node("scope", "scope"),
                        Node("scope/leaf", "record"),
                        Node("scope%2Fleaf", "record"),
                        Node("机械臂", "record"),
                    )
                )
            )
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED
            assert set(calls) == {
                run.run_id + "/scope/leaf",
                run.run_id + "/scope%2Fleaf",
                run.run_id + "/scope%252Fleaf",
                run.run_id + "/%E6%9C%BA%E6%A2%B0%E8%87%82",
            }
            assert run.operation_id("scope/leaf") == run.run_id + "/scope%2Fleaf"
            with pytest.raises(KeyError):
                run.operation_id("missing")

    asyncio.run(scenario())


def test_parent_feedback_wrapping_does_not_fail_valid_inner_progress():
    async def scenario():
        async def work(context, inputs):
            context.report({"ok": True})
            return inputs

        scoped = workflow_action(
            "scope",
            Workflow((Node("n", "work"),)),
            [Action("work", work)],
            max_feedback_bytes_per_report=32,
        )
        async with WorkflowEngine([scoped], max_feedback_bytes_per_report=2) as engine:
            run = engine.submit(Workflow((Node("scope", "scope"),)))
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


def test_inner_snapshot_failure_cannot_release_an_uncertain_scope():
    from unittest.mock import patch

    from orchestrion.workflow import WorkflowRun

    async def scenario():
        async def lost(context, inputs):
            raise ExecutionUncertain("controller unavailable")

        original = WorkflowRun.snapshot

        def broken_snapshot(run):
            if "lost" in run._results:
                raise RuntimeError("result observer failed")
            return original(run)

        scoped = workflow_action(
            "scope",
            Workflow((Node("lost", "lost"),)),
            [Action("lost", lost)],
            [Resource("arm")],
        )
        async with WorkflowEngine([scoped], [Resource("arm")]) as engine:
            with patch.object(WorkflowRun, "snapshot", broken_snapshot):
                run = engine.submit(Workflow((Node("scope", "scope"),)))
                assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            engine.confirm_quiescent(
                run.operation_id("scope"), evidence="whole scope verified stopped"
            )

    asyncio.run(scenario())
