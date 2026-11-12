"""Optional progress failures cannot affect device outcomes or strand queues."""

import asyncio
import subprocess
import sys
import threading

import pytest

from orchestrion.workflow import (
    Action,
    ActionContext,
    ActionOutcome,
    Node,
    Resource,
    ResourceClaim,
    RunStatus,
    Workflow,
    WorkflowEngine,
)


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_feedback_delivery_failure_preserves_unknown_outcome(
    threaded, error_type, monkeypatch
):
    async def scenario():
        async def device(context, inputs):
            if threaded:
                producer = threading.Thread(target=lambda: context.report({"progress": 1}))
                producer.start()
                producer.join(1)
                assert not producer.is_alive()
            else:
                context.report({"progress": 1})
            return ActionOutcome.unknown("device stop unconfirmed")

        async with WorkflowEngine(
            [Action("device", device, (ResourceClaim("arm"),))], [Resource("arm")]
        ) as engine:
            run = engine.submit(Workflow((Node("device", "device"),)))
            emit = run._emit

            def faulty_emit(node_id, kind, detail=None):
                if kind == "feedback":
                    raise error_type("progress sink unavailable")
                return emit(node_id, kind, detail)

            monkeypatch.setattr(run, "_emit", faulty_emit)
            result = await run.wait(1)
            assert result.status == RunStatus.NEEDS_ATTENTION
            assert result.nodes["device"].error == "device stop unconfirmed"
            assert result.nodes["device"].feedback_dropped == 1
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            engine.confirm_quiescent(run.operation_id("device"), evidence="controller idle")

    asyncio.run(scenario())


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize(
    "invalid_kind", ["nonfinite", "keys", "cycle", "budget", "cancelled"]
)
def test_invalid_feedback_logs_and_drops_without_changing_device_outcome(
    threaded, invalid_kind, caplog
):
    async def scenario():
        cycle = []
        cycle.append(cycle)

        class CancelledMapping(dict):
            def items(self):
                raise asyncio.CancelledError("progress snapshot cancelled")

        bad = {
            "nonfinite": {"value": float("nan")},
            "keys": {1: "bad key"},
            "cycle": cycle,
            "budget": {"image": "x" * 1000},
            "cancelled": CancelledMapping(value=1),
        }[invalid_kind]
        errors = []

        async def device(context, inputs):
            def report():
                try:
                    context.report(bad)
                    context.report({"ok": True})
                except BaseException as exc:
                    errors.append(exc)

            if threaded:
                producer = threading.Thread(target=report)
                producer.start()
                producer.join(1)
                assert not producer.is_alive()
            else:
                report()
            return ActionOutcome.unknown("device stop unconfirmed")

        async with WorkflowEngine(
            [Action("device", device, (ResourceClaim("arm"),))],
            [Resource("arm")],
            max_feedback_bytes_per_report=32,
        ) as engine:
            run = engine.submit(Workflow((Node("device", "device"),)))
            result = await run.wait(1)
            assert errors == []
            assert result.status == RunStatus.NEEDS_ATTENTION
            assert result.nodes["device"].error == "device stop unconfirmed"
            assert result.nodes["device"].feedback_dropped == 1
            assert [e.detail for e in run.events() if e.kind == "feedback"] == [
                {"ok": True}
            ]
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            assert "feedback was rejected" in caplog.text
            engine.confirm_quiescent(
                run.operation_id("device"), evidence="controller idle"
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_failed_sdk_notification_drops_accepted_reports_and_recovers(
    error_type, monkeypatch, caplog
):
    async def scenario():
        delivered, errors = [], []
        context = ActionContext("run", "device", delivered.append)
        loop = asyncio.get_running_loop()
        original_notify = loop.call_soon_threadsafe

        def broken_notify(callback, *args, **kwargs):
            raise error_type("notification unavailable")

        def produce(values):
            try:
                for value in values:
                    context.report(value)
            except BaseException as exc:
                errors.append(exc)

        try:
            with monkeypatch.context() as failure:
                failure.setattr(loop, "call_soon_threadsafe", broken_notify)
                thread = threading.Thread(target=produce, args=([0, 1, 2],))
                thread.start()
                thread.join(1)
                assert not thread.is_alive()
            assert not errors
            assert context._feedback_snapshot() == {"pending": 0, "dropped": 3}
            assert loop.call_soon_threadsafe == original_notify
            thread = threading.Thread(target=produce, args=([3],))
            thread.start()
            thread.join(1)
            assert not thread.is_alive()
            await asyncio.sleep(0)
            assert delivered == [3]
            assert context._finish() == 3
            assert sum("notification unavailable" in r.getMessage() for r in caplog.records) == 1
        finally:
            context._finish()

    asyncio.run(scenario())


def test_drain_callback_failure_counts_drops_and_delivers_remaining_progress(caplog):
    async def scenario():
        delivered, unhandled = [], []
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, detail: unhandled.append(detail))

        def deliver(value):
            if value in (1, 33):
                raise RuntimeError("progress callback failed")
            delivered.append(value)

        context = ActionContext("run", "device", deliver)
        try:
            producer = threading.Thread(target=lambda: [context.report(i) for i in range(64)])
            producer.start()
            producer.join(1)
            assert not producer.is_alive()
            for _ in range(4):
                await asyncio.sleep(0)
            assert delivered == [i for i in range(64) if i not in (1, 33)]
            assert context._finish() == 2
            assert not unhandled
            assert "progress callback failed" in caplog.text
        finally:
            context._finish()
            loop.set_exception_handler(original_handler)

    asyncio.run(scenario())


def test_reentrant_feedback_close_flushes_inflight_progress_in_order():
    async def scenario():
        delivered = []

        def deliver(value):
            delivered.append(value)
            if value == 0:
                context._finish()

        context = ActionContext("run", "device", deliver)
        try:
            thread = threading.Thread(target=lambda: [context.report(i) for i in range(64)])
            thread.start()
            thread.join(1)
            assert not thread.is_alive()
            await asyncio.sleep(0)
            assert delivered == list(range(64))
            assert context._feedback_snapshot() == {"pending": 0, "dropped": 0}
        finally:
            context._finish()

    asyncio.run(scenario())


def test_failed_drain_rescheduling_counts_lost_reports_and_can_resume(monkeypatch):
    async def scenario():
        delivered, unhandled = [], []
        context = ActionContext("run", "device", delivered.append)
        loop = asyncio.get_running_loop()
        original_schedule = loop.call_soon
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, detail: unhandled.append(detail))

        def produce(values):
            thread = threading.Thread(target=lambda: [context.report(i) for i in values])
            thread.start()
            thread.join(1)
            assert not thread.is_alive()

        try:
            produce(range(64))

            def fail_drain(callback, *args, **kwargs):
                if callback == context._feedback_channel._drain:
                    raise RuntimeError("drain scheduling unavailable")
                return original_schedule(callback, *args, **kwargs)

            with monkeypatch.context() as failure:
                failure.setattr(loop, "call_soon", fail_drain)
                await asyncio.sleep(0)
            assert delivered == list(range(32))
            assert context._feedback_snapshot() == {"pending": 0, "dropped": 32}
            produce([64])
            await asyncio.sleep(0)
            assert delivered == list(range(32)) + [64]
            assert context._finish() == 32
            assert not unhandled
        finally:
            context._finish()
            loop.set_exception_handler(original_handler)

    asyncio.run(scenario())


def test_overflow_callback_failure_does_not_lose_the_remaining_reports():
    async def scenario():
        delivered = []

        def overflow(count):
            raise RuntimeError("overflow observer unavailable")

        context = ActionContext(
            "run", "device", delivered.append,
            max_pending_feedback=2, feedback_overflow=overflow,
        )
        try:
            thread = threading.Thread(target=lambda: [context.report(i) for i in range(5)])
            thread.start()
            thread.join(1)
            assert not thread.is_alive()
            assert context._finish() == 3
            assert delivered == [3, 4]
        finally:
            context._finish()

    asyncio.run(scenario())


def test_retired_feedback_scalar_can_reenter_queue_status():
    probe = """
import asyncio
from orchestrion.workflow.feedback import FeedbackChannel
loop = asyncio.new_event_loop()
released = []
channel = FeedbackChannel(loop, lambda value: None, None, 1)
class Value(str):
    def __del__(self):
        channel.snapshot()
        released.append(str(self))
try:
    channel.post(Value('retired'))
    channel.post('current')
    assert released == ['retired']
finally:
    channel.close()
    loop.close()
print('retired feedback released without holding the queue lock')
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=3,
    )
    assert result.returncode == 0, result.stderr
    assert "without holding the queue lock" in result.stdout
