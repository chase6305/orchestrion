"""Device failures must remain observable even when SDK exceptions are broken."""

import concurrent.futures
import threading
import time

import pytest

from orchestrion import RetryPolicy
from orchestrion.tasks.function_call_task import CallableTask, InPlaceFunctionCallTask
from orchestrion.tasks.polling_task import PollingTask
from orchestrion.utils.logger import logger
from orchestrion.utils.types import PeekResponseResultType


class BrokenError(Exception):
    def __str__(self):
        raise ValueError("formatter failed")


def test_inline_failure_preserves_original_exception_and_records_response():
    class Task(InPlaceFunctionCallTask):
        def _call_fn(self, request_id, content):
            raise BrokenError()

    task = Task()
    with pytest.raises(BrokenError):
        task.invoke_async(1)
    response = task.peek_response(1)
    assert response.result_type == PeekResponseResultType.ErrorUnknown
    assert "BrokenError" in response.error


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


def test_failed_result_polling_logs_once_and_formats_once_per_read(caplog):
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
            assert len(formatted) == before + 10
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
