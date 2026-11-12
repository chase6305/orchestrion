"""Cancellation and execution budgets retain ownership until devices stop."""

import asyncio
import time

import pytest

from orchestrion.workflow import (
    Action,
    ActionCancelled,
    ExecutionUncertain,
    FailurePolicy,
    Node,
    NodeStatus,
    Resource,
    ResourceClaim,
    RunStatus,
    Workflow,
    WorkflowEngine,
)


async def echo(context, inputs):
    return inputs


async def reached(event):
    await asyncio.wait_for(event.wait(), 1)


@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"), reason="requires eager task support"
)
@pytest.mark.parametrize("topology", ["wide", "chain"])
@pytest.mark.parametrize("notification", ["ready", "timer"])
def test_eager_action_bursts_allow_scheduled_cancellation_before_more_device_calls(
    topology, notification
):
    async def scenario():
        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        calls, accepted = [], []
        count = 512

        async def device(context, inputs):
            calls.append(context.node_id)
            if len(calls) == 1:
                run = engine.get_run_by_key("burst")

                def cancel():
                    accepted.append(run.cancel())

                if notification == "timer":
                    loop.call_later(0, cancel)
                else:
                    loop.call_soon(cancel)

        engine = WorkflowEngine(
            [Action("device", device, (ResourceClaim("arm"),))],
            [Resource("arm")],
            max_concurrency=1,
        )
        try:
            loop.set_task_factory(asyncio.eager_task_factory)
            run = engine.submit(
                Workflow(
                    tuple(
                        Node(
                            str(index),
                            "device",
                            after=(
                                (str(index - 1),)
                                if topology == "chain" and index
                                else ()
                            ),
                        )
                        for index in range(count)
                    )
                ),
                idempotency_key="burst",
            )
            result = await run.wait(2)
            await asyncio.sleep(0)
            assert accepted == [True]
            assert 0 < len(calls) < count
            assert result.status is RunStatus.CANCELLED
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
            assert engine.scheduler_snapshot()["running_actions"] == 0
        finally:
            loop.set_task_factory(previous_factory)
            await engine.aclose(2)

    asyncio.run(scenario())


def test_workflow_budget_expires_while_waiting_for_another_runs_resource():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def hold(context, inputs):
            entered.set()
            await reached(release)

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [
                Action("hold", hold, (ResourceClaim("arm"),)),
                Action("record", record, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
        ) as engine:
            holder = engine.submit(Workflow((Node("hold", "hold"),)))
            try:
                await reached(entered)
                workflow = Workflow(
                    (
                        Node("queued", "record"),
                        Node("child", "record", after=("queued",)),
                    ),
                    timeout=0.01,
                )
                assert engine.validate(workflow)["timeout"] == 0.01
                run = engine.submit(workflow)
                result = await run.wait(1)
                assert result.status == RunStatus.FAILED
                assert result.nodes["queued"].status == NodeStatus.TIMED_OUT
                assert result.nodes["queued"].started_at is None
                assert result.nodes["child"].status == NodeStatus.TIMED_OUT
                assert calls == []
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            finally:
                release.set()
            await holder.wait(1)
            next_run = engine.submit(Workflow((Node("next", "record"),)))
            assert (await next_run.wait(1)).status == RunStatus.SUCCEEDED
            assert calls == ["next"]

    asyncio.run(scenario())


@pytest.mark.parametrize("uncertain", [False, True])
def test_workflow_timeout_drains_active_devices_and_quarantines_unknown_outcomes(
    uncertain,
):
    async def scenario():
        stopping, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def physical(context, inputs):
            assert await context.wait_cancelled() == "workflow_timeout"
            stopping.set()
            await reached(release)
            if uncertain:
                raise ExecutionUncertain("lost stop acknowledgement")
            raise ActionCancelled("controller is stationary")

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [
                Action("physical", physical, (ResourceClaim("base"),)),
                Action("record", record, (ResourceClaim("base"),)),
            ],
            [Resource("base")],
        ) as engine:
            run = engine.submit(Workflow((Node("move", "physical"),), timeout=0.01))
            next_run = engine.submit(Workflow((Node("next", "record"),)))
            try:
                await reached(stopping)
                assert run.snapshot().nodes["move"].status == NodeStatus.CANCELLING
                assert not run.done
                assert calls == []
                with pytest.raises(asyncio.TimeoutError):
                    await run.wait(0.01)
                assert engine.resource_snapshot()["resources"]["base"]["used"] == 1
            finally:
                release.set()
            result = await run.wait(1)
            if uncertain:
                assert result.status == RunStatus.NEEDS_ATTENTION
                with pytest.raises(asyncio.TimeoutError):
                    await next_run.wait(0.01)
                engine.confirm_quiescent(
                    run.run_id + "/move", evidence="controller query confirms stop"
                )
            else:
                assert result.nodes["move"].status == NodeStatus.TIMED_OUT
            assert (await next_run.wait(1)).status == RunStatus.SUCCEEDED
            assert calls == ["next"]

    asyncio.run(scenario())


@pytest.mark.parametrize("policy", list(FailurePolicy))
def test_total_budget_covers_dependencies_and_global_execution_queue(policy):
    async def scenario():
        async def work(context, inputs):
            await context.wait_cancelled()
            context.raise_if_cancelled()

        async with WorkflowEngine([Action("work", work)], max_concurrency=1) as engine:
            run = engine.submit(
                Workflow(
                    (
                        Node("active", "work"),
                        Node("queued", "work"),
                        Node("dependent", "work", after=("active",)),
                    ),
                    policy,
                    timeout=0.01,
                )
            )
            result = await run.wait(1)
            assert result.status == RunStatus.FAILED
            assert result.nodes["active"].status == NodeStatus.TIMED_OUT
            assert result.nodes["queued"].status == NodeStatus.TIMED_OUT
            assert result.nodes["dependent"].status == NodeStatus.TIMED_OUT

    asyncio.run(scenario())


def test_manual_cancel_keeps_its_reason_when_cleanup_outlasts_the_workflow_budget():
    async def scenario():
        entered = asyncio.Event()

        async def work(context, inputs):
            entered.set()
            assert await context.wait_cancelled() == "cancelled"
            await asyncio.sleep(0.03)
            return {"stopped": True}

        async with WorkflowEngine([Action("work", work)]) as engine:
            run = engine.submit(Workflow((Node("work", "work"),), timeout=0.02))
            await reached(entered)
            assert run.cancel()
            result = await run.wait(1)
            assert result.status == RunStatus.CANCELLED
            assert result.nodes["work"].status == NodeStatus.CANCELLED

    asyncio.run(scenario())


def test_node_timeout_stops_fail_fast_admission_before_device_cleanup_completes():
    async def scenario():
        stopping, release = asyncio.Event(), asyncio.Event()

        async def hold(context, inputs):
            assert await context.wait_cancelled() == "timeout"
            stopping.set()
            await reached(release)
            context.raise_if_cancelled()

        async with WorkflowEngine([Action("hold", hold)], max_concurrency=1) as engine:
            run = engine.submit(
                Workflow(
                    (
                        Node("timeout", "hold", timeout=0.01),
                        Node("queued", "hold"),
                    )
                )
            )
            try:
                await reached(stopping)
                assert run.snapshot().nodes["queued"].status == NodeStatus.SKIPPED
                assert not run.done
            finally:
                release.set()
            assert (await run.wait(1)).nodes["timeout"].status == NodeStatus.TIMED_OUT

    asyncio.run(scenario())


@pytest.mark.parametrize("scope", ["workflow", "node"])
def test_delayed_timer_does_not_report_success_after_elapsed_budget(scope):
    async def scenario():
        async def stalled_adapter(context, inputs):
            # Deliberately stall the loop to model an incorrectly blocking adapter.
            # The software cannot force-stop it, but must report the missed budget.
            time.sleep(0.02)
            return {"applied": True}

        async with WorkflowEngine([Action("stall", stalled_adapter)]) as engine:
            run = engine.submit(
                Workflow(
                    (
                        Node(
                            "stall", "stall", timeout=0.001 if scope == "node" else None
                        ),
                    ),
                    timeout=0.001 if scope == "workflow" else None,
                )
            )
            result = await run.wait(1)
            assert result.nodes["stall"].status == NodeStatus.TIMED_OUT
            assert result.nodes["stall"].output == {"applied": True}

    asyncio.run(scenario())


def test_delayed_workflow_timer_expires_before_new_resource_reservations():
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [Action("record", record, (ResourceClaim("arm"),))], [Resource("arm")]
        ) as engine:
            run = engine.submit(
                Workflow(
                    (Node("first", "record"), Node("second", "record")), timeout=0.001
                )
            )
            # The driver is ready before the due timer. Neither may run until the
            # caller yields, after this deliberately elapsed workflow budget.
            time.sleep(0.01)
            result = await run.wait(1)
            assert result.status is RunStatus.FAILED
            assert all(
                node.status is NodeStatus.TIMED_OUT for node in result.nodes.values()
            )
            assert all(node.started_at is None for node in result.nodes.values())
            assert calls == []
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
            followup = engine.submit(Workflow((Node("next", "record"),)))
            assert (await followup.wait(1)).status is RunStatus.SUCCEEDED
            assert calls == ["next"]

    asyncio.run(scenario())


def test_successful_run_cancels_its_budget_timer():
    async def scenario():
        async def echo(context, inputs):
            return inputs

        async with WorkflowEngine([Action("echo", echo)]) as engine:
            run = engine.submit(Workflow((Node("echo", "echo"),), timeout=0.02))
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED
            assert run._deadline_timer.cancelled()
            await asyncio.sleep(0.03)
            assert run.snapshot().status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [False, 0, -1, float("nan"), float("inf"), "1"])
def test_invalid_total_budgets_are_rejected(timeout):
    with pytest.raises(ValueError):
        Workflow((Node("echo", "echo"),), timeout=timeout)


@pytest.mark.parametrize("deadline", [False, True])
def test_cancel_and_timeout_hold_resources_until_adapter_stops(deadline):
    async def scenario():
        entered, stopping, release, next_entered = (asyncio.Event() for _ in range(4))

        async def physical(context, inputs):
            entered.set()
            reason = await context.wait_cancelled()
            assert reason == ("timeout" if deadline else "cancelled")
            stopping.set()
            await reached(release)
            raise ActionCancelled("device confirmed stationary")

        async def next_action(context, inputs):
            next_entered.set()

        async with WorkflowEngine(
            [
                Action("move", physical, (ResourceClaim("arm"),)),
                Action("next", next_action, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
        ) as engine:
            first = engine.submit(
                Workflow((Node("move", "move", timeout=0.02 if deadline else None),))
            )
            await reached(entered)
            second = engine.submit(Workflow((Node("next", "next"),)))
            if not deadline:
                assert first.cancel() and not first.cancel()
            await reached(stopping)
            assert first.snapshot().nodes["move"].status == NodeStatus.CANCELLING
            assert not first.done and not next_entered.is_set()
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            with pytest.raises(asyncio.TimeoutError):
                await first.wait(0.01)
            release.set()
            result = await first.wait(1)
            assert result.nodes["move"].status == (
                NodeStatus.TIMED_OUT if deadline else NodeStatus.CANCELLED
            )
            await second.wait(1)
            assert next_entered.is_set()

    asyncio.run(scenario())


def test_wait_timeout_and_cancelled_waiter_do_not_cancel_execution():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        contexts = []

        async def physical(context, inputs):
            contexts.append(context)
            entered.set()
            await reached(release)

        async with WorkflowEngine([Action("move", physical)]) as engine:
            run = engine.submit(Workflow((Node("move", "move"),)))
            await reached(entered)
            with pytest.raises(asyncio.TimeoutError):
                await run.wait(0.01)
            waiter = asyncio.create_task(run.wait())
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert not contexts[0].cancel_requested
            release.set()
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


def test_cancel_before_dispatch_never_calls_adapter_and_engine_releases_runs():
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [Action("work", record)], max_active_runs=1
        ) as engine:
            first = engine.submit(Workflow((Node("one", "work"),)))
            with pytest.raises(RuntimeError, match="max_active_runs"):
                engine.submit(Workflow((Node("two", "work"),)))
            first.cancel()
            assert (await first.wait(1)).status == RunStatus.CANCELLED
            assert calls == []
            second = engine.submit(Workflow((Node("two", "work"),)))
            assert (await second.wait(1)).status == RunStatus.SUCCEEDED
            assert not engine._runs
            assert calls == ["two"]

    asyncio.run(scenario())


@pytest.mark.parametrize("queued_count", [0, 32])
def test_close_timeout_preserves_ownership_and_can_be_awaited_again(queued_count):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def move(context, inputs):
            entered.set()
            await reached(release)

        async def record(context, inputs):
            calls.append(context.node_id)

        engine = WorkflowEngine(
            [
                Action("move", move, (ResourceClaim("arm"),)),
                Action("claimed", record, (ResourceClaim("arm"),)),
                Action("free", record),
            ],
            [Resource("arm")],
            max_concurrency=1,
        )
        run = engine.submit(Workflow((Node("move", "move"),)))
        try:
            await reached(entered)
            queued = [
                engine.submit(
                    Workflow((Node("queued", "claimed" if index % 2 else "free"),))
                )
                for index in range(queued_count)
            ]
            with pytest.raises(asyncio.TimeoutError):
                await engine.aclose(0.01)
            for pending in queued:
                assert (await pending.wait(1)).status == RunStatus.CANCELLED
                assert pending.snapshot().nodes["queued"].started_at is None
            assert calls == []
            assert not run.done
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            with pytest.raises(RuntimeError, match="closed"):
                engine.submit(Workflow((Node("next", "move"),)))
        finally:
            release.set()
            await engine.aclose(1)
        assert run.done
        assert (await run.wait(1)).status == RunStatus.CANCELLED

    asyncio.run(scenario())


def test_action_timeout_excludes_time_waiting_for_a_resource():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold(context, inputs):
            entered.set()
            await reached(release)

        async with WorkflowEngine(
            [
                Action("hold", hold, (ResourceClaim("arm"),)),
                Action("echo", echo, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
        ) as engine:
            first = engine.submit(Workflow((Node("hold", "hold"),)))
            await reached(entered)
            second = engine.submit(Workflow((Node("next", "echo", timeout=0.01),)))
            with pytest.raises(asyncio.TimeoutError):
                await second.wait(0.03)
            assert second.snapshot().nodes["next"].status == NodeStatus.READY
            release.set()
            await first.wait(1)
            assert (await second.wait(1)).status == RunStatus.SUCCEEDED

    asyncio.run(scenario())
