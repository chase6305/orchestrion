"""Engine and adapter lifecycle, task startup, and scheduler failure cleanup."""

import asyncio
import gc
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from orchestrion.workflow import (
    Action,
    BlockingAction,
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


@pytest.mark.parametrize("timeout", [None, 1e-9, 1.0])
def test_completed_wait_preserves_isolated_results_and_loop_checks_after_close(timeout):
    async def scenario():
        async def finish(context, inputs):
            context.checkpoint({"ticket": [1]})
            return {"applied": [True]}

        async with WorkflowEngine([Action("finish", finish)]) as engine:
            run = engine.submit(Workflow((Node("finish", "finish"),)))
            await run.wait(1)
        result = await run.wait(timeout)
        assert result.status == RunStatus.SUCCEEDED
        result.nodes["finish"].output["applied"].clear()
        result.nodes["finish"].checkpoint["ticket"].clear()
        fresh = await run.wait(timeout)
        assert fresh.nodes["finish"].output == {"applied": [True]}
        assert fresh.nodes["finish"].checkpoint == {"ticket": [1]}
        with pytest.raises(ValueError):
            await run.wait(0)
        return run

    run = asyncio.run(scenario())

    async def foreign_loop():
        with pytest.raises(RuntimeError, match="original event loop"):
            await run.wait(timeout)

    asyncio.run(foreign_loop())


def test_late_cancel_does_not_rewrite_a_completed_success():
    async def scenario():
        cancellations = []

        async def finish(context, inputs):
            asyncio.get_running_loop().call_soon(
                lambda: cancellations.append(run.cancel())
            )
            return {"applied": True}

        async with WorkflowEngine([Action("finish", finish)]) as engine:
            run = engine.submit(Workflow((Node("finish", "finish"),)))
            result = await run.wait(1)
            assert cancellations == [False]
            assert result.status == RunStatus.SUCCEEDED
            assert result.nodes["finish"].status == NodeStatus.SUCCEEDED

    asyncio.run(scenario())


def test_action_cannot_wait_on_its_own_run_through_a_child_coroutine():
    async def scenario():
        async def recursive_wait(context, inputs):
            with pytest.raises(RuntimeError, match="own WorkflowEngine"):
                await asyncio.create_task(run.wait(0.02))
            return {"ok": True}

        async with WorkflowEngine([Action("wait", recursive_wait)]) as engine:
            run = engine.submit(Workflow((Node("wait", "wait"),)))
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["_schedule", "_execute"])
def test_task_factory_failure_rolls_back_admission_or_releases_claims(stage):
    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        rejected = []
        calls = []

        def factory(loop, coroutine, **kwargs):
            if coroutine.cr_code.co_name == stage and not rejected:
                rejected.append(coroutine)
                raise RuntimeError("injected task factory failure")
            return asyncio.Task(coroutine, loop=loop, **kwargs)

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [
                Action("record", record, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
        ) as engine:
            loop.set_task_factory(factory)
            try:
                if stage == "_schedule":
                    with pytest.raises(RuntimeError, match="injected task factory"):
                        engine.submit(Workflow((Node("bad", "record"),)))
                    assert not engine._runs
                else:
                    run = engine.submit(
                        Workflow(
                            (
                                Node("bad", "record"),
                                Node("good", "record"),
                            ),
                            FailurePolicy.CONTINUE_INDEPENDENT,
                        )
                    )
                    result = await run.wait(1)
                    assert result.nodes["bad"].status == NodeStatus.FAILED
                    assert result.nodes["good"].status == NodeStatus.SUCCEEDED
                assert rejected[0].cr_frame is None
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
            finally:
                loop.set_task_factory(original_factory)
            next_run = engine.submit(Workflow((Node("next", "record"),)))
            assert (await next_run.wait(1)).status == RunStatus.SUCCEEDED
            assert "bad" not in calls

    asyncio.run(scenario())


def test_worker_cancelled_before_start_does_not_leave_a_lease_or_pending_run():
    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        calls = []

        def factory(loop, coroutine, **kwargs):
            task = asyncio.Task(coroutine, loop=loop, **kwargs)
            if coroutine.cr_code.co_name == "_execute":
                task.cancel()
            return task

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [
                Action("record", record, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
        ) as engine:
            loop.set_task_factory(factory)
            try:
                run = engine.submit(Workflow((Node("cancelled", "record"),)))
                assert (await run.wait(1)).nodes[
                    "cancelled"
                ].status == NodeStatus.CANCELLED
                assert not engine.resource_snapshot()["quarantined"]
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
                assert not calls
            finally:
                loop.set_task_factory(original_factory)

    asyncio.run(scenario())


def test_malformed_sdk_exception_does_not_strand_the_workflow():
    async def scenario():
        class MalformedError(Exception):
            def __str__(self):
                raise ValueError("formatter failed")

        async def fail(context, inputs):
            raise MalformedError()

        async with WorkflowEngine(
            [
                Action("fail", fail, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
        ) as engine:
            run = engine.submit(Workflow((Node("fail", "fail"),)))
            result = await run.wait(1)
            assert result.status == RunStatus.FAILED
            assert (
                result.nodes["fail"].error
                == "MalformedError: exception message is unavailable"
            )
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0

    asyncio.run(scenario())


def test_abrupt_worker_failure_retains_uncertain_device_resources():
    async def scenario():
        class AdapterAbort(BaseException):
            pass

        async def abort(context, inputs):
            raise AdapterAbort("device state was not confirmed")

        async with WorkflowEngine(
            [
                Action("abort", abort, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
        ) as engine:
            run = engine.submit(Workflow((Node("abort", "abort"),)))
            result = await run.wait(1)
            assert result.status == RunStatus.NEEDS_ATTENTION
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            engine.confirm_quiescent(
                run.run_id + "/abort", evidence="controller confirmed stop"
            )

    asyncio.run(scenario())


def test_completed_context_cannot_wait_forever_or_start_a_new_sdk_command():
    async def scenario():
        contexts, calls = [], []

        async def capture(context, inputs):
            contexts.append(context)

        async with WorkflowEngine([Action("capture", capture)]) as engine:
            run = engine.submit(Workflow((Node("capture", "capture"),)))
            await run.wait(1)
            context = contexts[0]
            with pytest.raises(RuntimeError, match="already completed"):
                await context.wait_cancelled()
            with pytest.raises(RuntimeError, match="already completed"):
                await context.sleep(0.01)
            with pytest.raises(RuntimeError, match="already completed"):
                await BlockingAction(lambda inputs: calls.append(inputs))(context, {})
            assert calls == []

    asyncio.run(scenario())


def test_interrupted_executor_wait_prevents_queued_sdk_io_and_observes_late_failure():
    async def scenario():
        loop = asyncio.get_running_loop()
        queued = asyncio.Event()
        release = threading.Event()
        calls, unobserved = [], []
        original_handler = loop.get_exception_handler()

        class RecordingExecutor(ThreadPoolExecutor):
            def submit(self, function, *args, **kwargs):
                future = super().submit(function, *args, **kwargs)
                if getattr(self, "observe", False):
                    queued.set()
                return future

        def sdk(inputs):
            calls.append(inputs)
            raise RuntimeError("SDK should not have started")

        loop.set_exception_handler(lambda loop, details: unobserved.append(details))
        try:
            with RecordingExecutor(max_workers=1) as executor:
                executor.submit(release.wait, 2)
                executor.observe = True
                async with WorkflowEngine(
                    [
                        Action(
                            "sdk",
                            BlockingAction(sdk, executor),
                            (ResourceClaim("arm"),),
                        ),
                    ],
                    [Resource("arm")],
                    stop_timeout=0.005,
                ) as engine:
                    try:
                        run = engine.submit(Workflow((Node("sdk", "sdk"),)))
                        await asyncio.wait_for(queued.wait(), 1)
                        next(iter(engine._workers)).cancel()
                        assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
                        assert (
                            engine.resource_snapshot()["resources"]["arm"]["used"] == 1
                        )
                    finally:
                        release.set()
                    await asyncio.wait_for(
                        asyncio.wrap_future(executor.submit(lambda: None)), 1
                    )
                    await engine.aclose(1)
                    assert not calls
                    engine.confirm_quiescent(
                        run.run_id + "/sdk",
                        evidence="executor drained; SDK never started",
                    )
                gc.collect()
                await asyncio.sleep(0)
                assert not unobserved
        finally:
            loop.set_exception_handler(original_handler)

    asyncio.run(scenario())


def test_executor_shutdown_cancels_queued_calls_without_quarantining_the_device():
    async def scenario():
        queued = asyncio.Event()
        release = threading.Event()
        calls = []

        class RecordingExecutor(ThreadPoolExecutor):
            def submit(self, function, *args, **kwargs):
                future = super().submit(function, *args, **kwargs)
                if getattr(self, "observe", False):
                    queued.set()
                return future

        with RecordingExecutor(max_workers=1) as executor:
            executor.submit(release.wait, 2)
            executor.observe = True
            async with WorkflowEngine(
                [
                    Action(
                        "sdk",
                        BlockingAction(lambda inputs: calls.append(inputs), executor),
                        (ResourceClaim("arm"),),
                    ),
                ],
                [Resource("arm")],
            ) as engine:
                try:
                    run = engine.submit(Workflow((Node("sdk", "sdk"),)))
                    await asyncio.wait_for(queued.wait(), 1)
                    executor.shutdown(wait=False, cancel_futures=True)
                    result = await run.wait(1)
                    assert result.nodes["sdk"].status == NodeStatus.CANCELLED
                    assert not engine.resource_snapshot()["quarantined"]
                    assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
                    assert not calls
                finally:
                    release.set()

    asyncio.run(scenario())


def test_interrupted_blocking_action_keeps_live_sdk_in_draining_accounting():
    async def scenario():
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), threading.Event()
        finished = threading.Event()
        calls = []

        def sdk(inputs):
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(3)
                calls.append("device stopped")
            finally:
                finished.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            engine = WorkflowEngine(
                [Action("sdk", BlockingAction(sdk, executor), (ResourceClaim("arm"),))],
                [Resource("arm")], stop_timeout=0.005,
            )
            try:
                run = engine.submit(Workflow((Node("sdk", "sdk"),)))
                await asyncio.wait_for(entered.wait(), 1)
                worker = next(iter(engine._workers))
                worker.cancel()
                assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
                snapshot = engine.scheduler_snapshot()
                assert snapshot["draining_adapters"] == 1
                assert snapshot["running_actions"] == 1
                assert snapshot["resources"]["arm"]["used"] == 1
                with pytest.raises(RuntimeError, match="still draining"):
                    engine.confirm_quiescent(run.operation_id("sdk"), evidence="controller idle")
                worker.cancel()
                with pytest.raises(asyncio.TimeoutError):
                    await engine.aclose(0.005)
                assert not finished.is_set()
            finally:
                release.set()
                await engine.aclose(1)
            assert finished.is_set() and calls == ["device stopped"]
            assert engine.scheduler_snapshot()["draining_adapters"] == 0
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            engine.confirm_quiescent(run.operation_id("sdk"), evidence="SDK drained; controller idle")

    asyncio.run(scenario())


@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"), reason="requires eager task support"
)
def test_eager_task_factory_preserves_lifecycle_and_resource_accounting():
    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        calls = []

        async def immediate(context, inputs):
            calls.append(context.node_id)
            if context.node_id == "bad":
                raise ValueError("injected failure")
            return inputs

        async with WorkflowEngine(
            [
                Action("immediate", immediate, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
        ) as engine:
            loop.set_task_factory(asyncio.eager_task_factory)
            try:
                run = engine.submit(
                    Workflow(
                        (
                            Node("first", "immediate"),
                            Node("next", "immediate", after=("first",)),
                        ),
                        timeout=1,
                    )
                )
                assert (await run.wait(1)).status == RunStatus.SUCCEEDED
                failed = engine.submit(
                    Workflow(
                        (
                            Node("bad", "immediate"),
                            Node("skipped", "immediate"),
                        )
                    )
                )
                assert (await failed.wait(1)).status == RunStatus.FAILED
                assert calls == ["first", "next", "bad"]
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
            finally:
                loop.set_task_factory(original_factory)

    asyncio.run(scenario())


def test_cancelled_blocking_call_does_not_start_after_executor_queue_wait():
    async def scenario():
        queued = asyncio.Event()
        release = threading.Event()
        calls = []

        class RecordingExecutor(ThreadPoolExecutor):
            def submit(self, function, *args, **kwargs):
                result = super().submit(function, *args, **kwargs)
                if getattr(self, "observe", False):
                    queued.set()
                return result

        def sdk(inputs):
            calls.append(inputs)
            return {"applied": True}

        with RecordingExecutor(max_workers=1) as executor:
            blocker = executor.submit(release.wait, 2)
            executor.observe = True
            async with WorkflowEngine(
                [
                    Action(
                        "sdk", BlockingAction(sdk, executor), (ResourceClaim("arm"),)
                    ),
                ],
                [Resource("arm")],
            ) as engine:
                try:
                    run = engine.submit(Workflow((Node("sdk", "sdk"),)))
                    await asyncio.wait_for(queued.wait(), 1)
                    assert run.cancel()
                finally:
                    release.set()
                result = await run.wait(1)
                assert result.nodes["sdk"].status == NodeStatus.CANCELLED
                assert calls == []
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
                assert blocker.result(1)

    asyncio.run(scenario())


def test_action_cannot_close_its_own_engine_and_deadlock_on_itself():
    async def scenario():
        outcomes = []

        async def close(context, inputs):
            try:
                await engine.aclose(0.02)
            except Exception as exc:
                outcomes.append(type(exc))
            return {"closed": engine._closed}

        async with WorkflowEngine([Action("close", close)]) as engine:
            run = engine.submit(Workflow((Node("close", "close"),)))
            result = await run.wait(1)
            assert outcomes == [RuntimeError]
            assert result.status == RunStatus.SUCCEEDED
            assert result.nodes["close"].output == {"closed": False}

    asyncio.run(scenario())


def test_queued_worker_observes_scheduler_failure_before_entering_adapter():
    async def scenario():
        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        spawned, calls = [], []

        def factory(loop, coroutine, **kwargs):
            if coroutine.cr_code.co_name == "_execute":
                if spawned:
                    raise asyncio.CancelledError("second worker could not start")
                spawned.append(True)
            return asyncio.Task(coroutine, loop=loop, **kwargs)

        async def record(context, inputs):
            calls.append(context.node_id)

        engine = WorkflowEngine([Action("record", record)])
        loop.set_task_factory(factory)
        try:
            run = engine.submit(
                Workflow((Node("first", "record"), Node("second", "record")))
            )
            result = await run.wait(1)
            assert result.status == RunStatus.FAILED
            assert all(
                node.status == NodeStatus.FAILED for node in result.nodes.values()
            )
            assert not calls
            assert engine.scheduler_snapshot()["running_actions"] == 0
        finally:
            loop.set_task_factory(previous_factory)
        await engine.aclose(1)

    asyncio.run(scenario())


@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"), reason="requires eager task support"
)
def test_eager_reentrant_replay_does_not_confuse_previous_completed_driver():
    async def scenario():
        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        observed = []

        async def replay(context, inputs):
            key = inputs["cycle"]
            repeated = engine.submit(workflows[key], idempotency_key=key)
            observed.append(repeated.run_id)

        workflows = {
            key: Workflow((Node("n", "replay", {"cycle": key}),))
            for key in ("first", "second")
        }
        async with WorkflowEngine([Action("replay", replay)]) as engine:
            loop.set_task_factory(asyncio.eager_task_factory)
            try:
                for key, workflow in workflows.items():
                    run = engine.submit(workflow, idempotency_key=key)
                    assert (await run.wait(1)).status == RunStatus.SUCCEEDED
                assert len(set(observed)) == 2
                assert engine.scheduler_snapshot()["scheduler_error"] is None
            finally:
                loop.set_task_factory(previous_factory)

    asyncio.run(scenario())


def test_driver_cancelled_during_worker_startup_releases_unstarted_reservation():
    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        calls = []

        def factory(loop, coroutine, **kwargs):
            if coroutine.cr_code.co_name == "_execute":
                raise asyncio.CancelledError("interrupted task startup")
            return asyncio.Task(coroutine, loop=loop, **kwargs)

        async def record(context, inputs):
            calls.append(context.node_id)

        engine = WorkflowEngine(
            [Action("record", record, (ResourceClaim("arm"),))], [Resource("arm")]
        )
        loop.set_task_factory(factory)
        try:
            run = engine.submit(Workflow((Node("n", "record"),)))
            assert (await run.wait(1)).status == RunStatus.FAILED
            assert not calls
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
            assert not engine.resource_snapshot()["quarantined"]
            assert engine.scheduler_snapshot()["running_actions"] == 0
        finally:
            loop.set_task_factory(original_factory)
        await engine.aclose(1)

    asyncio.run(scenario())


def test_driver_cancelled_before_start_finishes_run_and_retains_keyed_result():
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)

        engine = WorkflowEngine([Action("record", record)])
        run = engine.submit(Workflow((Node("n", "record"),)), idempotency_key="cycle")
        engine._driver.cancel()
        result = await run.wait(1)
        assert result.status == RunStatus.FAILED
        assert result.nodes["n"].error == "scheduler_failed"
        assert not calls
        assert engine.get_run_by_key("cycle") is run
        snapshot = engine.scheduler_snapshot()
        assert snapshot["closed"] is True
        assert "interrupted" in snapshot["scheduler_error"]
        assert snapshot["active_runs"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            engine.submit(Workflow((Node("n", "record"),)))
        await engine.aclose(1)
        await engine.aclose(1)

    asyncio.run(scenario())


@pytest.mark.parametrize("uncertain", [False, True])
@pytest.mark.parametrize("queued_count", [0, 8])
def test_driver_failure_drains_active_actions_and_never_starts_queued_work(
    uncertain, queued_count
):
    async def scenario():
        entered, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        async def move(context, inputs):
            calls.append(context.node_id)
            entered.set()
            assert await context.wait_cancelled() == "scheduler_failed"
            stopping.set()
            await asyncio.wait_for(release.wait(), 1)
            if uncertain:
                raise ExecutionUncertain("controller state unavailable")
            context.raise_if_cancelled()

        engine = WorkflowEngine(
            [
                Action("move", move, (ResourceClaim("arm"),)),
                Action("echo", echo),
                Action("claimed", echo, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
            max_concurrency=1,
        )
        run = engine.submit(
            Workflow(
                (
                    Node("move", "move"),
                    Node("queued", "echo"),
                    Node("dependent", "echo", after=("move",)),
                )
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            queued = [
                engine.submit(
                    Workflow((Node("queued", "claimed" if index % 2 else "echo"),))
                )
                for index in range(queued_count)
            ]
            engine._driver.cancel()
            await asyncio.wait_for(stopping.wait(), 1)
            for pending in queued:
                assert (await pending.wait(1)).status == RunStatus.FAILED
                assert pending.snapshot().nodes["queued"].started_at is None
            assert run.snapshot().nodes["move"].status == NodeStatus.CANCELLING
            assert run.snapshot().nodes["queued"].status == NodeStatus.FAILED
            assert run.snapshot().nodes["dependent"].status == NodeStatus.FAILED
            with pytest.raises(asyncio.TimeoutError):
                await engine.aclose(0.001)
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            assert not run.done
        finally:
            release.set()
        result = await run.wait(1)
        assert calls == ["move"]
        assert result.status == (
            RunStatus.NEEDS_ATTENTION if uncertain else RunStatus.FAILED
        )
        await engine.aclose(1)
        assert engine.resource_snapshot()["resources"]["arm"]["used"] == int(uncertain)
        if uncertain:
            engine.confirm_quiescent(
                run.run_id + "/move", evidence="controller stopped"
            )
        assert engine.scheduler_snapshot()["running_actions"] == 0
        assert not engine._ready_heap

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["exception", "early_return"])
def test_unexpected_driver_exit_is_observed_without_unhandled_task_exception(kind):
    async def scenario():
        errors = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))

        class BrokenEngine(WorkflowEngine):
            async def _schedule(self):
                if kind == "exception":
                    raise RuntimeError("injected scheduler failure")

        engine = BrokenEngine([Action("echo", echo)])
        try:
            run = engine.submit(Workflow((Node("n", "echo"),)))
            assert (await run.wait(1)).status == RunStatus.FAILED
            error = engine.scheduler_snapshot()["scheduler_error"]
            assert ("injected" if kind == "exception" else "active runs") in error
            await engine.aclose(1)
            assert not errors
        finally:
            loop.set_exception_handler(previous_handler)

    asyncio.run(scenario())


def test_submit_detects_failed_driver_before_its_completion_callback_runs():
    async def scenario():
        class BrokenEngine(WorkflowEngine):
            async def _schedule(self):
                raise RuntimeError("injected failure")

        engine = BrokenEngine([Action("echo", echo)])
        run = engine.submit(Workflow((Node("n", "echo"),)))
        await asyncio.sleep(0)
        assert engine._driver.done()
        assert not run.done  # The driver's done callback is still queued.
        with pytest.raises(RuntimeError, match="closed"):
            engine.submit(Workflow((Node("second", "echo"),)))
        assert (await run.wait(1)).status == RunStatus.FAILED
        await engine.aclose(1)

    asyncio.run(scenario())


def test_cancelling_close_waiter_does_not_cancel_cleanup_or_driver():
    async def scenario():
        entered, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def move(context, inputs):
            entered.set()
            await context.wait_cancelled()
            stopping.set()
            await asyncio.wait_for(release.wait(), 1)
            context.raise_if_cancelled()

        engine = WorkflowEngine([Action("move", move)])
        run = engine.submit(Workflow((Node("move", "move"),)))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            closer = asyncio.create_task(engine.aclose())
            await asyncio.wait_for(stopping.wait(), 1)
            closer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closer
            assert not engine._driver.done()
            assert not run.done
        finally:
            release.set()
        await engine.aclose(1)
        assert (await run.wait(1)).status == RunStatus.CANCELLED
        assert engine.scheduler_snapshot()["scheduler_error"] is None

    asyncio.run(scenario())
