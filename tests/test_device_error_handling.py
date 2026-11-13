"""Device failures must remain observable even when SDK exceptions are broken."""

import concurrent.futures
import threading
import time
import weakref

import pytest

from orchestrion import RetryPolicy, TaskFailure
from orchestrion.tasks.function_call_task import CallableTask, InPlaceFunctionCallTask
from orchestrion.tasks.polling_task import PollingTask
from orchestrion.utils.logger import logger
from orchestrion.utils.types import PeekResponseResultType


class BrokenError(Exception):
    def __str__(self):
        raise ValueError("formatter failed")


def test_inline_failure_records_response_without_raising(caplog):
    class Task(InPlaceFunctionCallTask):
        def _call_fn(self, request_id, content):
            raise BrokenError()

    task = Task()
    assert task.invoke_async(1)
    response = task.peek_response(1)
    assert response.result_type == PeekResponseResultType.ErrorUnknown
    assert "BrokenError" in response.error
    assert response.error in caplog.text


@pytest.mark.parametrize("max_attempts", [1, 2])
def test_retry_survives_broken_exception_formatter(max_attempts):
    attempts = []

    def call(request_id, content):
        attempts.append(request_id)
        if len(attempts) == 1:
            raise BrokenError()
        return {"ok": True}

    task = CallableTask(call, retry_policy=RetryPolicy(max_attempts=max_attempts, delay=0))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    task.initialize(executor=executor)
    try:
        assert task.invoke_async(1)
        deadline = time.monotonic() + 1
        while task.peek_response(1).result_type == PeekResponseResultType.RequestSentNoResponse:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        response = task.peek_response(1)
        if max_attempts == 2:
            assert response.content == {"ok": True}
        else:
            assert response.result_type == PeekResponseResultType.ErrorUnknown
            assert "BrokenError" in response.error
        assert attempts == [1] * max_attempts
    finally:
        task.stop()
        executor.shutdown(wait=True)


def test_polling_recovers_from_broken_exception_formatter():
    recovered = threading.Event()
    attempts = []

    def poll():
        attempts.append(None)
        if len(attempts) == 1:
            raise BrokenError()
        recovered.set()
        return {}

    task = PollingTask(poll, interval=0.001)
    task.initialize()
    try:
        assert recovered.wait(1)
    finally:
        task.stop()


def test_sample_copy_does_not_hold_status_lock():
    class Sample(dict):
        def __deepcopy__(self, memo):
            assert task._sample_lock.acquire(timeout=0.1)
            task._sample_lock.release()
            copied.set()
            return {"values": [1]}

    copied = threading.Event()
    task = PollingTask(lambda: Sample(), interval=1)
    task.initialize()
    try:
        assert copied.wait(1)
    finally:
        task.stop()
    first = task._call_fn(1, None)
    first["sample"]["values"].append(2)
    assert task._call_fn(2, None)["sample"] == {"values": [1]}


@pytest.mark.parametrize("has_cached_sample", [False, True])
def test_invalid_poll_copy_preserves_cache_and_recovers(has_cached_sample):
    class InvalidSample(dict):
        def __deepcopy__(self, memo):
            return ["not a dictionary"]

    failed, recover, recovered = threading.Event(), threading.Event(), threading.Event()
    initial_samples = [{"values": [1]}] if has_cached_sample else []

    def poll():
        if initial_samples:
            return initial_samples.pop()
        return {"values": [2]} if recover.is_set() else InvalidSample()

    task = PollingTask(poll, interval=0.005)

    def on_sample():
        status = task.peek_status()
        if status["last_error"]:
            failed.set()
        elif recover.is_set():
            recovered.set()

    task.set_completion_callback(on_sample)
    task.initialize()
    try:
        assert failed.wait(1)
        status = task.peek_status()
        assert status["health"] == ("degraded" if has_cached_sample else "offline")
        assert status["available"] is has_cached_sample
        assert status["consecutive_failures"] >= 1
        assert "copied poll sample" in status["last_error"]
        if has_cached_sample:
            assert task.invoke_async(0)
            assert task.peek_response(0).content["sample"] == {"values": [1]}
        else:
            assert task.invoke_async(0)
            assert (
                task.peek_response(0).result_type is PeekResponseResultType.ErrorUnknown
            )
            assert "copied poll sample" in task.peek_response(0).error

        recover.set()
        assert recovered.wait(1)
        status = task.peek_status()
        assert status["health"] == "online"
        assert status["consecutive_failures"] == 0
        assert status["last_error"] is None
        assert task.invoke_async(1)
        assert task.peek_response(1).content["sample"] == {"values": [2]}
    finally:
        task.stop()


def test_failed_result_polling_logs_and_formats_once(caplog):
    formatted = []
    completed = threading.Event()

    class DeviceError(Exception):
        def __str__(self):
            assert task._lock.acquire(timeout=0.1)
            task._lock.release()
            formatted.append(True)
            return "device rejected command"

    def call(request_id, content):
        raise DeviceError()

    task = CallableTask(call)
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0)
            assert completed.wait(1)
            before = len(formatted)
            for _ in range(10):
                response = task.peek_response(0)
                assert response.error == "device rejected command"
            assert len(formatted) == before == 1
            messages = [
                record.getMessage() for record in caplog.records
                if record.getMessage().startswith("Request 0 failed:")
            ]
            assert messages == ["Request 0 failed: device rejected command"]
        finally:
            task.stop()


def test_logging_failure_does_not_suppress_completion_notification(monkeypatch):
    completed = threading.Event()

    def broken_log(*args, **kwargs):
        raise RuntimeError("log sink unavailable")

    def call(request_id, content):
        raise ValueError("device rejected command")

    monkeypatch.setattr(logger, "error", broken_log)
    task = CallableTask(call)
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0)
            assert completed.wait(1)
            assert task.peek_response(0).error == "device rejected command"
            assert task.peek_status()["failed"] == 1
        finally:
            task.stop()


@pytest.mark.parametrize("threaded", [False, True])
def test_returned_task_failure_is_retained_and_next_request_can_succeed(
    threaded, caplog
):
    def call(request_id, content):
        return TaskFailure("interlock open") if request_id == 0 else content

    class InlineTask(InPlaceFunctionCallTask):
        def _call_fn(self, request_id, content):
            return call(request_id, content)

    task = CallableTask(call) if threaded else InlineTask()
    completed = threading.Event()
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0)
            assert completed.wait(1)
            response = task.peek_response(0)
            assert response.result_type is PeekResponseResultType.ErrorUnknown
            assert response.error == "interlock open"
            assert "Request 0 failed: interlock open" in caplog.text
            if threaded:
                assert task._future_map[0].exception(timeout=1) is None
                assert task.peek_status()["failed"] == 1
            completed.clear()
            assert task.invoke_async(1, {"ok": True})
            assert completed.wait(1)
            assert task.peek_response(1).content == {"ok": True}
        finally:
            task.stop()


def test_cooperative_cancellation_is_a_returned_outcome_without_retry():
    completed = threading.Event()
    calls = []

    def call(request_id, content):
        calls.append(request_id)
        return TaskFailure("controller confirmed stop", cancelled=True)

    task = CallableTask(call, retry_policy=RetryPolicy(max_attempts=3))
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0)
            assert completed.wait(1)
            assert task._future_map[0].exception(timeout=1) is None
            status = task.peek_status()
            assert status["cancelled"] == 1
            assert status["failed"] == status["pending"] == status["total_retries"] == 0
            assert task.peek_response(0).error == "controller confirmed stop"
            assert calls == [0]
        finally:
            task.stop()


def test_inline_completion_and_logging_failures_do_not_escape(monkeypatch):
    def broken(*args, **kwargs):
        raise BrokenError()

    class Task(InPlaceFunctionCallTask):
        def _call_fn(self, request_id, content):
            return TaskFailure("device offline")

    task = Task()
    task.set_completion_callback(broken)
    monkeypatch.setattr(logger, "error", broken)
    assert task.invoke_async(0)
    assert task.peek_response(0).error == "device offline"
    assert not task.invoke_async([])


def test_polling_logs_changed_failures_once_and_logs_again_after_recovery(caplog):
    samples = iter(
        [
            {},
            ValueError("offline"),
            ValueError("offline"),
            ValueError("bad sample"),
            {},
            ValueError("offline"),
        ]
    )

    def poll():
        value = next(samples)
        if isinstance(value, Exception):
            raise value
        return value

    task = PollingTask(poll, interval=0.001)
    completed = threading.Event()
    notifications = []

    def next_poll():
        notifications.append(True)
        if len(notifications) == 6:
            task._poll_stop.set()
            completed.set()

    task.set_completion_callback(next_poll)
    task.initialize()
    try:
        assert completed.wait(1)
    finally:
        task.stop()
    messages = [
        r.getMessage() for r in caplog.records if "Device polling failed" in r.message
    ]
    assert messages == [
        "Device polling failed: offline",
        "Device polling failed: bad sample",
        "Device polling failed: offline",
    ]


def test_completed_failure_does_not_retain_sdk_traceback_payload():
    reclaimed, completed = threading.Event(), threading.Event()
    references = []

    class SDKPayload:
        pass

    def call(request_id, content):
        payload = SDKPayload()
        references.append(weakref.ref(payload, lambda ref: reclaimed.set()))
        raise RuntimeError("SDK failed")

    task = CallableTask(call)
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0)
            assert completed.wait(1)
            assert task.peek_response(0).error == "SDK failed"
            assert task._future_map[0].exception(timeout=1) is None
            assert reclaimed.wait(1)
            assert references[0]() is None
        finally:
            task.stop()
