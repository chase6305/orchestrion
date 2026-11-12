"""Submission replay and explicit retention of device execution outcomes."""

import asyncio
import gc
import weakref
from dataclasses import replace

import pytest

from orchestrion.workflow import (
    Action,
    ExecutionUncertain,
    FailurePolicy,
    Node,
    OutputRef,
    Resource,
    ResourceClaim,
    RunStatus,
    Workflow,
    WorkflowEngine,
)


async def echo(context, inputs):
    return inputs


@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"), reason="requires eager task support"
)
def test_key_is_reserved_before_eager_action_start_and_reentrant_replay():
    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        observed = []

        async def immediate(context, inputs):
            replay = engine.submit(workflow, idempotency_key="cycle")
            observed.append(replay.run_id)
            assert engine.get_run_by_key("cycle") is replay
            assert engine.prune_idempotency_keys(0) == 0

        workflow = Workflow((Node("n", "immediate"),))
        async with WorkflowEngine([Action("immediate", immediate)]) as engine:
            loop.set_task_factory(asyncio.eager_task_factory)
            try:
                run = engine.submit(workflow, idempotency_key="cycle")
                assert (await run.wait(1)).status == RunStatus.SUCCEEDED
                assert observed == [run.run_id]
                assert engine.submit(workflow, idempotency_key="cycle") is run
            finally:
                loop.set_task_factory(original_factory)

    asyncio.run(scenario())


def test_replay_returns_original_active_and_completed_run_at_admission_limit():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def move(context, inputs):
            calls.append(context.operation_id)
            entered.set()
            await asyncio.wait_for(release.wait(), 1)
            return {"reached": inputs["target"]}

        workflow = Workflow((Node("move", "move", {"target": [1, 2, 3]}),))
        engine = WorkflowEngine(
            [Action("move", move)], max_active_runs=1, max_idempotency_keys=1
        )
        async with engine:
            try:
                first = engine.submit(workflow, idempotency_key="cycle-42")
                assert engine.submit(workflow, idempotency_key="cycle-42") is first
                await asyncio.wait_for(entered.wait(), 1)
                with pytest.raises(asyncio.TimeoutError):
                    await first.wait(0.001)
                assert engine.submit(workflow, idempotency_key="cycle-42") is first
                with pytest.raises(RuntimeError, match="max_idempotency_keys"):
                    engine.submit(workflow, idempotency_key="cycle-43")
                assert engine.prune_idempotency_keys(0) == 0
                assert first.idempotency_key == "cycle-42"
                with pytest.raises(AttributeError):
                    first.idempotency_key = "changed"
            finally:
                release.set()
            assert (await first.wait(1)).status == RunStatus.SUCCEEDED
            assert engine.submit(workflow, idempotency_key="cycle-42") is first
            assert calls == [first.run_id + "/move"]
            result = first.snapshot()
            result.nodes["move"].output["reached"].clear()
            assert first.snapshot().nodes["move"].output == {"reached": [1, 2, 3]}
        assert engine.get_run_by_key("cycle-42") is first
        assert engine.get_run_by_key("absent") is None
        with pytest.raises(RuntimeError, match="closed"):
            engine.submit(workflow, idempotency_key="cycle-42")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    [
        "inputs",
        "action",
        "priority",
        "node_timeout",
        "claims",
        "dependencies",
        "order",
        "policy",
        "workflow_timeout",
    ],
)
def test_same_key_rejects_changed_execution_snapshot_before_device_io(change):
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)

        nodes = (Node("a", "record", {"target": 1}), Node("b", "record"))
        original = Workflow(nodes)
        node_changes = {
            "inputs": {"inputs": {"target": 2}},
            "action": {"action": "other"},
            "priority": {"priority": 1},
            "node_timeout": {"timeout": 1},
            "claims": {"claims": (ResourceClaim("arm", 2),)},
            "dependencies": {"after": ("b",)},
        }
        changed = original
        if change in node_changes:
            changed = replace(
                original, nodes=(replace(nodes[0], **node_changes[change]), nodes[1])
            )
        elif change == "order":
            changed = replace(original, nodes=tuple(reversed(nodes)))
        elif change == "policy":
            changed = replace(
                original, failure_policy=FailurePolicy.CONTINUE_INDEPENDENT
            )
        elif change == "workflow_timeout":
            changed = replace(original, timeout=1)
        async with WorkflowEngine(
            [Action("record", record), Action("other", record)], [Resource("arm", 2)]
        ) as engine:
            first = engine.submit(original, idempotency_key="cycle")
            with pytest.raises(ValueError, match="different workflow"):
                engine.submit(changed, idempotency_key="cycle")
            assert not calls
            await first.wait(1)
            assert calls == ["a", "b"]

    asyncio.run(scenario())


@pytest.mark.parametrize("first,second", [(True, 1), (1, 1.0), (0.0, -0.0)])
def test_input_fingerprint_preserves_json_scalar_types(first, second):
    async def scenario():
        async with WorkflowEngine([Action("echo", echo)]) as engine:
            run = engine.submit(
                Workflow((Node("n", "echo", {"value": first}),)), idempotency_key="key"
            )
            with pytest.raises(ValueError, match="different workflow"):
                engine.submit(
                    Workflow((Node("n", "echo", {"value": second}),)),
                    idempotency_key="key",
                )
            await run.wait(1)

    asyncio.run(scenario())


def test_literal_preview_marker_cannot_impersonate_an_output_reference():
    async def scenario():
        source = Node("source", "echo", {"pose": [1]})
        bound = Workflow(
            (source, Node("move", "echo", {"x": OutputRef("source", ("pose",))}))
        )
        literal = Workflow(
            (
                source,
                Node(
                    "move",
                    "echo",
                    {"x": {"$output": {"node_id": "source", "path": ["pose"]}}},
                    after=("source",),
                ),
            )
        )
        async with WorkflowEngine([Action("echo", echo)]) as engine:
            assert engine.validate(bound) == engine.validate(literal)
            run = engine.submit(bound, idempotency_key="key")
            with pytest.raises(ValueError, match="different workflow"):
                engine.submit(literal, idempotency_key="key")
            assert (await run.wait(1)).nodes["move"].output == {"x": [1]}

    asyncio.run(scenario())


def test_fingerprint_uses_copied_inputs_and_normalized_claims():
    async def scenario():
        original = Workflow((Node("n", "echo", {"a": [1], "b": 2}),))
        equivalent = Workflow(
            (Node("n", "echo", {"b": 2, "a": [1]}, claims=(ResourceClaim("arm"),)),)
        )
        async with WorkflowEngine(
            [Action("echo", echo, (ResourceClaim("arm"),))], [Resource("arm")]
        ) as engine:
            run = engine.submit(original, idempotency_key="key")
            original.nodes[0].inputs["a"].append(99)
            assert engine.submit(equivalent, idempotency_key="key") is run
            with pytest.raises(ValueError, match="different workflow"):
                engine.submit(original, idempotency_key="key")
            assert (await run.wait(1)).nodes["n"].output == {"a": [1], "b": 2}

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["failed", "cancelled", "timed_out"])
def test_terminal_failure_replays_result_without_retry(outcome):
    async def scenario():
        calls = []

        async def work(context, inputs):
            calls.append(context.operation_id)
            if outcome == "failed":
                raise ValueError("device rejected command")
            await context.sleep(1)

        workflow = Workflow(
            (Node("n", "work"),), timeout=0.001 if outcome == "timed_out" else None
        )
        async with WorkflowEngine([Action("work", work)]) as engine:
            run = engine.submit(workflow, idempotency_key="cycle")
            if outcome == "cancelled":
                run.cancel()
            first = await run.wait(1)
            count = len(calls)
            replay = engine.submit(workflow, idempotency_key="cycle")
            assert replay is run
            assert await replay.wait(1) == first
            assert len(calls) == count
            assert first.status == (
                RunStatus.CANCELLED if outcome == "cancelled" else RunStatus.FAILED
            )

    asyncio.run(scenario())


def test_pruning_preserves_active_and_each_unreconciled_operation():
    async def scenario():
        calls = []

        async def lost(context, inputs):
            calls.append(context.operation_id)
            raise ExecutionUncertain("connection lost")

        workflow = Workflow(
            (Node("a", "lost"), Node("b", "lost")), FailurePolicy.CONTINUE_INDEPENDENT
        )
        async with WorkflowEngine([Action("lost", lost)]) as engine:
            run = engine.submit(workflow, idempotency_key="cycle")
            assert engine.prune_idempotency_keys(0) == 0
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert engine.prune_idempotency_keys(0) == 0
            assert engine.submit(workflow, idempotency_key="cycle") is run
            engine.confirm_quiescent(run.run_id + "/a", evidence="device a stopped")
            assert engine.prune_idempotency_keys(0) == 0
            engine.confirm_quiescent(run.run_id + "/b", evidence="device b stopped")
            assert engine.submit(workflow, idempotency_key="cycle") is run
            assert engine.prune_idempotency_keys(0) == 1
            assert engine.get_run_by_key("cycle") is None
            assert run.snapshot().status == RunStatus.NEEDS_ATTENTION
            assert len(calls) == 2

    asyncio.run(scenario())


def test_explicit_pruning_releases_capacity_and_allows_intentional_key_reuse():
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.operation_id)
            return inputs

        workflow = Workflow((Node("n", "record"),))
        async with WorkflowEngine(
            [Action("record", record)], max_idempotency_keys=2
        ) as engine:
            first = engine.submit(workflow, idempotency_key="first")
            await first.wait(1)
            second = engine.submit(workflow, idempotency_key="second")
            await second.wait(1)
            with pytest.raises(RuntimeError, match="max_idempotency_keys"):
                engine.submit(workflow, idempotency_key="third")
            assert engine.prune_idempotency_keys() == 0
            assert engine.prune_idempotency_keys(keep_last=2) == 0
            assert engine.get_run_by_key("first") is first
            assert engine.get_run_by_key("second") is second
            assert engine.prune_idempotency_keys(keep_last=1) == 1
            assert engine.get_run_by_key("first") is None
            assert engine.get_run_by_key("second") is second
            assert (await first.wait(1)).status == RunStatus.SUCCEEDED
            replacement = engine.submit(workflow, idempotency_key="first")
            assert replacement is not first
            await replacement.wait(1)
            assert len(calls) == 3
            assert len(set(calls)) == 3

    asyncio.run(scenario())


def test_failed_submission_does_not_reserve_a_key_or_exhaust_retention():
    async def scenario():
        engine = WorkflowEngine([Action("echo", echo)], max_idempotency_keys=1)
        loop = asyncio.get_running_loop()
        factory = loop.get_task_factory()

        def reject_driver(loop, coroutine, **kwargs):
            if coroutine.cr_code.co_name == "_schedule":
                raise RuntimeError("cannot start scheduler")
            return asyncio.Task(coroutine, loop=loop, **kwargs)

        async with engine:
            with pytest.raises(ValueError, match="unknown action"):
                engine.submit(Workflow((Node("n", "missing"),)), idempotency_key="key")
            assert engine.get_run_by_key("key") is None
            loop.set_task_factory(reject_driver)
            try:
                with pytest.raises(RuntimeError, match="cannot start scheduler"):
                    engine.submit(Workflow((Node("n", "echo"),)), idempotency_key="key")
            finally:
                loop.set_task_factory(factory)
            assert engine.get_run_by_key("key") is None
            run = engine.submit(Workflow((Node("n", "echo"),)), idempotency_key="key")
            await run.wait(1)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_at", [1, 2])
def test_ready_queue_preparation_failure_releases_admission_and_allows_key_reuse(
    failure_at, monkeypatch
):
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)

        workflow = Workflow(
            (
                Node("first", "record"),
                Node("second", "record"),
                Node("dependent", "record", after=("second",)),
            )
        )
        engine = WorkflowEngine(
            [Action("record", record, (ResourceClaim("arm"),))], [Resource("arm")]
        )
        attempts = []
        prepare = engine._make_ready

        def fail_preparation(run, item):
            prepare(run, item)
            attempts.append(item.node.node_id)
            if len(attempts) == failure_at:
                raise RuntimeError("ready queue preparation failed")

        try:
            with monkeypatch.context() as failure:
                failure.setattr(engine, "_make_ready", fail_preparation)
                with pytest.raises(
                    RuntimeError, match="ready queue preparation failed"
                ):
                    engine.submit(workflow, idempotency_key="retry")
            assert calls == []
            assert engine.get_run_by_key("retry") is None
            snapshot = engine.scheduler_snapshot()
            assert snapshot["active_runs"] == 0
            assert snapshot["running_actions"] == 0
            assert snapshot["resources"]["arm"]["waiting"] == 0
            assert snapshot["resources"]["arm"]["used"] == 0
            retry = engine.submit(workflow, idempotency_key="retry")
            assert (await retry.wait(1)).status is RunStatus.SUCCEEDED
            assert calls == ["first", "second", "dependent"]
        finally:
            await engine.aclose(1)

    asyncio.run(scenario())


def test_unkeyed_submissions_are_not_deduplicated_or_retained():
    async def scenario():
        engine = WorkflowEngine([Action("echo", echo)], max_idempotency_keys=1)
        workflow = Workflow((Node("n", "echo"),))
        async with engine:
            retained = engine.submit(workflow, idempotency_key="key")
            first = engine.submit(workflow)
            second = engine.submit(workflow)
            assert first is not second
            assert first.idempotency_key is None
            await asyncio.gather(retained.wait(1), first.wait(1), second.wait(1))
            reference = weakref.ref(first)
            del first
            # Let worker and driver completion callbacks relinquish their handles.
            await asyncio.sleep(0)
            gc.collect()
            assert reference() is None
            assert engine.get_run_by_key("key") is retained

    asyncio.run(scenario())


@pytest.mark.parametrize("key", ["", 1, False, [], {}])
def test_invalid_idempotency_keys_are_rejected(key):
    async def scenario():
        async with WorkflowEngine([Action("echo", echo)]) as engine:
            with pytest.raises((TypeError, ValueError)):
                engine.submit(Workflow((Node("n", "echo"),)), idempotency_key=key)
            with pytest.raises((TypeError, ValueError)):
                engine.get_run_by_key(key)

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [-1, True, 1.5, None])
def test_invalid_retention_limits_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        WorkflowEngine([], max_idempotency_keys=value)

    async def scenario():
        async with WorkflowEngine([]) as engine:
            with pytest.raises(ValueError):
                engine.prune_idempotency_keys(value)

    asyncio.run(scenario())
