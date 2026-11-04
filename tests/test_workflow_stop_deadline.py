"""Stop acknowledgement budgets never release unconfirmed physical resources."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from orchestrion.workflow import (
    Action,
    BlockingAction,
    Node,
    NodeStatus,
    Resource,
    ResourceClaim,
    RunStatus,
    SQLiteJournal,
    Workflow,
    WorkflowEngine,
)


@pytest.mark.parametrize("budget_on_node", [False, True])
def test_stop_deadline_records_unknown_while_adapter_and_resources_remain_live(
    budget_on_node,
):
    async def scenario():
        entered = asyncio.Event()
        stopping = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def hold(context, inputs):
            entered.set()
            await context.wait_cancelled()
            stopping.set()
            await release.wait()
            context.report({"late": True})
            return {"stopped": True}

        async def next_step(context, inputs):
            calls.append(context.node_id)

        engine = WorkflowEngine(
            [
                Action("hold", hold, (ResourceClaim("arm"),)),
                Action("next", next_step, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
            stop_timeout=None if budget_on_node else 0.01,
        )
        async with engine:
            run = engine.submit(
                Workflow(
                    (
                        Node(
                            "hold",
                            "hold",
                            stop_timeout=0.01 if budget_on_node else None,
                        ),
                    )
                )
            )
            next_run = engine.submit(Workflow((Node("next", "next"),)))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                run.cancel()
                await asyncio.wait_for(stopping.wait(), 1)
                result = await run.wait(1)
                assert result.status == RunStatus.NEEDS_ATTENTION
                assert result.nodes["hold"].status == NodeStatus.UNKNOWN
                assert "stop confirmation deadline" in result.nodes["hold"].error
                assert not calls and not next_run.done
                snapshot = engine.scheduler_snapshot()
                assert snapshot["draining_adapters"] == 1
                assert snapshot["running_actions"] == 1
                assert snapshot["resources"]["arm"]["used"] == 1
                with pytest.raises(RuntimeError, match="still draining"):
                    engine.confirm_quiescent(
                        run.run_id + "/hold", evidence="controller idle"
                    )
            finally:
                release.set()
            for _ in range(20):
                if engine.scheduler_snapshot()["draining_adapters"] == 0:
                    break
                await asyncio.sleep(0)
            assert engine.scheduler_snapshot()["running_actions"] == 0
            assert run.snapshot().nodes["hold"].status == NodeStatus.UNKNOWN
            assert not calls
            assert not any(event.detail == {"late": True} for event in run.events())
            assert any(event.kind == "adapter_drained" for event in run.events())
            engine.confirm_quiescent(
                run.run_id + "/hold", evidence="adapter exited and controller idle"
            )
            assert (await next_run.wait(1)).status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


def test_shutdown_still_waits_for_unknown_adapters_and_uses_one_timeout_budget(
    tmp_path,
):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def hold(context, inputs):
            entered.set()
            await release.wait()
            return {"stopped": True}

        journal = SQLiteJournal(tmp_path / "cell.db")
        engine = WorkflowEngine(
            [Action("hold", hold, (ResourceClaim("arm"),))],
            [Resource("arm")],
            stop_timeout=0.01,
            journal=journal,
        )
        try:
            run = engine.submit(Workflow((Node("hold", "hold"),)))
            await asyncio.wait_for(entered.wait(), 1)
            with pytest.raises(asyncio.TimeoutError):
                await engine.aclose(0.03)
            assert run.done
            assert run.snapshot().status == RunStatus.NEEDS_ATTENTION
            assert journal.get_run(run.run_id)["nodes"][0]["lease_active"]
            with pytest.raises(RuntimeError, match="drain"):
                journal.close()
        finally:
            release.set()
            await engine.aclose(1)
            journal.close()
        assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1

    asyncio.run(scenario())


def test_unknown_adapter_retains_concurrency_slot_until_it_really_exits():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold(context, inputs):
            entered.set()
            await release.wait()

        async def ready(context, inputs):
            return inputs

        async with WorkflowEngine(
            [Action("hold", hold), Action("ready", ready)],
            max_concurrency=1,
            stop_timeout=0.01,
        ) as engine:
            run = engine.submit(Workflow((Node("hold", "hold"),)))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                run.cancel()
                assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
                queued = engine.submit(Workflow((Node("ready", "ready"),)))
                with pytest.raises(asyncio.TimeoutError):
                    await queued.wait(0.01)
                assert queued.explain("ready")["blockers"] == [
                    {"kind": "concurrency", "running": 1, "limit": 1}
                ]
            finally:
                release.set()
            assert (await queued.wait(1)).status == RunStatus.SUCCEEDED
            engine.confirm_quiescent(run.run_id + "/hold", evidence="handler drained")

    asyncio.run(scenario())


def test_prompt_stop_confirmation_cancels_deadline_and_does_not_quarantine():
    async def scenario():
        entered = asyncio.Event()

        async def hold(context, inputs):
            entered.set()
            await context.wait_cancelled()
            context.raise_if_cancelled()

        async with WorkflowEngine([Action("hold", hold)], stop_timeout=0.1) as engine:
            run = engine.submit(Workflow((Node("hold", "hold"),)))
            await asyncio.wait_for(entered.wait(), 1)
            run.cancel()
            assert (await run.wait(1)).status == RunStatus.CANCELLED
            assert not engine._stop_timers
            assert not engine.resource_snapshot()["quarantined"]

    asyncio.run(scenario())


def test_blocking_sdk_still_drains_after_stop_budget_without_thread_cancellation():
    async def scenario():
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def sdk(inputs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(2)
            return {"finished": True}

        with ThreadPoolExecutor(1) as executor:
            async with WorkflowEngine(
                [Action("sdk", BlockingAction(sdk, executor), (ResourceClaim("arm"),))],
                [Resource("arm")],
                stop_timeout=0.01,
            ) as engine:
                run = engine.submit(Workflow((Node("sdk", "sdk"),)))
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                    run.cancel()
                    assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
                    assert engine.scheduler_snapshot()["draining_adapters"] == 1
                finally:
                    release.set()
            assert engine.scheduler_snapshot()["draining_adapters"] == 0
            engine.confirm_quiescent(
                run.run_id + "/sdk", evidence="SDK completed and device idle"
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "1"])
def test_invalid_stop_budgets_rejected_before_dispatch(value):
    with pytest.raises(ValueError):
        Node("n", "hold", stop_timeout=value)
    with pytest.raises(ValueError):
        WorkflowEngine([], stop_timeout=value)


def test_blocked_loop_still_respects_elapsed_stop_confirmation_deadline():
    import time

    async def scenario():
        entered = asyncio.Event()

        async def uncooperative(context, inputs):
            entered.set()
            await context.wait_cancelled()
            time.sleep(0.02)
            return {"stopped": True}

        async with WorkflowEngine(
            [Action("work", uncooperative)], stop_timeout=0.005
        ) as engine:
            run = engine.submit(Workflow((Node("work", "work"),)))
            await asyncio.wait_for(entered.wait(), 1)
            run.cancel()
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert run.snapshot().nodes["work"].status == NodeStatus.UNKNOWN
            await engine.aclose(1)
            engine.confirm_quiescent(
                run.run_id + "/work", evidence="late adapter has drained"
            )

    asyncio.run(scenario())
