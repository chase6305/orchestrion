import concurrent.futures
import threading
import time

import pytest

from orchestrion import RetryPolicy
from orchestrion.tasks.function_call_task import (
    CallableTask,
    InPlaceFunctionCallTask,
    ThreadedPoolFunctionCallTask,
)
from orchestrion.tasks.polling_task import PollingTask
from orchestrion.utils.types import PeekResponseResultType


# Subclass for testing, override _call_fn
class InPlaceTaskStub(InPlaceFunctionCallTask):

    def _call_fn(self, request_id, content):
        # Simply return content
        return {"result": content["value"] * 2}


class ThreadedPoolTaskStub(ThreadedPoolFunctionCallTask):

    def _call_fn(self, request_id, content):
        # Simulate a time-consuming operation
        time.sleep(0.1)
        return {"result": content["value"] + 1}


@pytest.mark.parametrize("task_type", [InPlaceTaskStub, ThreadedPoolTaskStub])
@pytest.mark.parametrize("value", [0, 1.5, True])
def test_invalid_max_result_count_is_rejected(task_type, value):
    expected = TypeError if value in (1.5, True) else ValueError
    with pytest.raises(expected):
        task_type(max_result_count=value)


def test_task_inplace():
    task = InPlaceTaskStub()
    assert task.invoke_async(1, {"value": 10})
    resp = task.peek_response(1)
    print("InPlace resp:", resp.content)
    assert resp.content["result"] == 20


def test_failing_completion_callback_does_not_fail_request():
    task = InPlaceTaskStub()

    def fail_callback():
        raise RuntimeError("callback failure")

    task.set_completion_callback(fail_callback)
    assert task.invoke_async(1, {"value": 10})
    assert task.peek_response(1).result_type is PeekResponseResultType.ResponseFound


def test_task_threaded():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        task = ThreadedPoolTaskStub()
        task.initialize(executor=executor)
        assert task.invoke_async(1, {"value": 5})
        # Immediately peek, should not be finished
        resp = task.peek_response(1)
        print("Threaded first peek:", resp)
        time.sleep(0.2)
        # Peek again, should be finished
        resp = task.peek_response(1)
        print("Threaded resp:", resp.content)
        assert resp.content["result"] == 6
        status = task.peek_status()
        assert status["pending"] == 0
        assert status["succeeded"] == 1
        assert status["failed"] == 0


def test_callable_task_adapts_call_and_status_functions():
    task = CallableTask(
        lambda request_id, content: {
            "request_id": request_id,
            "value": content["value"] * 2,
        },
        status=lambda: {"device_type": "camera", "frames": 4},
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        assert task.invoke_async(3, {"value": 5})
        deadline = time.monotonic() + 0.5
        while task.peek_response(3).result_type is not PeekResponseResultType.ResponseFound:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert task.peek_response(3).content == {"request_id": 3, "value": 10}
        assert task.peek_status()["device_type"] == "camera"


def test_callable_task_description_is_static_and_defensive():
    metadata = {"device_type": "camera", "commands": ["capture"]}
    task = CallableTask(
        lambda request_id, content: content,
        retry_policy=RetryPolicy(
            max_attempts=2, retry_exceptions=(ConnectionError,)
        ),
        metadata=metadata,
    )
    metadata["commands"].append("mutated")
    description = task.describe()
    assert description["kind"] == "command"
    assert description["execution"] == "thread_pool"
    assert description["capabilities"] == {
        "cancellation": True,
        "response_pruning": True,
        "status": True,
    }
    assert description["metadata"]["commands"] == ["capture"]
    assert description["retry"]["max_attempts"] == 2
    assert description["retry"]["exception_types"] == ["ConnectionError"]
    description["metadata"].clear()
    assert task.describe()["metadata"]["device_type"] == "camera"


def test_task_metadata_must_be_json_compatible():
    with pytest.raises(ValueError, match="JSON-compatible"):
        CallableTask(lambda request_id, content: content, metadata={"bad": {1, 2}})


def test_callable_task_validates_callbacks_and_status_result():
    with pytest.raises(TypeError, match="call must"):
        CallableTask(None)
    with pytest.raises(TypeError, match="status must"):
        CallableTask(lambda request_id, content: content, status=object())

    task = CallableTask(lambda request_id, content: content, status=lambda: "bad")
    with pytest.raises(TypeError, match="status callable"):
        task.peek_status()


def test_callable_task_retries_selected_transient_failures():
    attempts = []

    def flaky_call(request_id, content):
        attempts.append(request_id)
        if len(attempts) < 3:
            raise ConnectionError("camera reconnecting")
        return {"ok": True}

    task = CallableTask(
        flaky_call,
        retry_policy=RetryPolicy(
            max_attempts=3,
            delay=0,
            retry_exceptions=(ConnectionError,),
        ),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        assert task.invoke_async(1)
        deadline = time.monotonic() + 0.5
        while task.peek_response(1).result_type is not PeekResponseResultType.ResponseFound:
            assert time.monotonic() < deadline
            time.sleep(0.001)
    assert attempts == [1, 1, 1]
    assert task.peek_status()["total_retries"] == 2
    assert task.peek_status()["latest_attempts"] == 3


def test_callable_task_does_not_retry_unselected_failure():
    task = CallableTask(
        lambda request_id, content: (_ for _ in ()).throw(ValueError("bad command")),
        retry_policy=RetryPolicy(
            max_attempts=3, retry_exceptions=(ConnectionError,)
        ),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        assert task.invoke_async(1)
        deadline = time.monotonic() + 0.5
        response = task.peek_response(1)
        while response.result_type is not PeekResponseResultType.ErrorUnknown:
            assert time.monotonic() < deadline
            time.sleep(0.001)
            response = task.peek_response(1)
        assert response.error == "bad command"
    assert task.peek_status()["total_retries"] == 0


def test_callable_task_stop_interrupts_retry_backoff():
    attempted = threading.Event()

    def unavailable(request_id, content):
        attempted.set()
        raise ConnectionError("offline")

    task = CallableTask(
        unavailable,
        retry_policy=RetryPolicy(
            max_attempts=3,
            delay=30.0,
            retry_exceptions=(ConnectionError,),
        ),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        assert task.invoke_async(1)
        assert attempted.wait(0.5)
        started = time.monotonic()
        task.stop()
        deadline = started + 0.5
        while task.peek_response(1).result_type is not PeekResponseResultType.ErrorUnknown:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert time.monotonic() - started < 0.5


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"max_attempts": 0}, ValueError),
        ({"max_attempts": True}, TypeError),
        ({"delay": -1}, ValueError),
        ({"backoff": 0}, ValueError),
        ({"max_delay": float("inf")}, ValueError),
        ({"retry_exceptions": []}, TypeError),
        ({"retry_exceptions": ()}, ValueError),
    ],
)
def test_retry_policy_validates_configuration(kwargs, expected):
    with pytest.raises(expected):
        RetryPolicy(**kwargs)


def test_retry_policy_caps_overflowing_exponential_delay():
    policy = RetryPolicy(delay=1.0, backoff=1e308, max_delay=5.0)
    assert policy.delay_before(4) == 5.0


def test_polling_task_caches_latest_telemetry_and_reports_health():
    sample_ready = threading.Event()
    counter = {"value": 0}

    def poll():
        counter["value"] += 1
        sample_ready.set()
        return {"temperature": 20 + counter["value"]}

    task = PollingTask(poll, interval=0.005)
    task.initialize()
    try:
        assert sample_ready.wait(0.5)
        assert task.invoke_async(1)
        response = task.peek_response(1)
        assert response.result_type is PeekResponseResultType.ResponseFound
        assert response.content["sample"]["temperature"] >= 21
        status = task.peek_status()
        assert status["health"] == "online"
        assert status["available"]
        assert status["observed_at"] > 0
    finally:
        task.stop()
    assert task.peek_status()["health"] == "offline"
    assert not task.peek_status()["initialized"]


def test_polling_task_description_identifies_cached_telemetry():
    task = PollingTask(
        lambda: {"celsius": 24.0},
        interval=0.25,
        metadata={"unit": "celsius"},
    )
    description = task.describe()
    assert description["kind"] == "telemetry"
    assert description["execution"] == "background_polling"
    assert description["poll_interval"] == 0.25
    assert description["metadata"] == {"unit": "celsius"}


def test_polling_task_degrades_after_failure_but_keeps_last_sample():
    second_poll = threading.Event()
    attempts = {"count": 0}

    def poll():
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {"input": True}
        second_poll.set()
        raise ConnectionError("PLC timeout")

    task = PollingTask(poll, interval=0.005)
    task.initialize()
    try:
        assert second_poll.wait(0.5)
        deadline = time.monotonic() + 0.5
        while task.peek_status()["consecutive_failures"] == 0:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        status = task.peek_status()
        assert status["health"] == "degraded"
        assert status["available"]
        assert status["last_error"] == "PLC timeout"
        assert task.invoke_async(1)
        assert task.peek_response(1).content["sample"] == {"input": True}
    finally:
        task.stop()


@pytest.mark.parametrize("value", [0, -1, float("nan"), True])
def test_polling_task_validates_interval(value):
    with pytest.raises(ValueError, match="interval"):
        PollingTask(lambda: {}, interval=value)


def test_task_inplace_duplicate_request():
    task = InPlaceTaskStub()
    assert task.invoke_async(1, {"value": 3})
    # Duplicate request_id should return False
    assert not task.invoke_async(1, {"value": 4})


def test_out_of_order_request_does_not_mark_gap_as_flushed():
    task = InPlaceTaskStub()
    assert task.invoke_async(2, {"value": 2})
    assert (
        task.peek_response(1).result_type
        is PeekResponseResultType.ErrorRequestNotSent
    )
    assert task.invoke_async(0, {"value": 0})
    assert task.invoke_async(1, {"value": 1})
    assert not task.invoke_async(2, {"value": 2})


def test_task_inplace_max_result_count():
    task = InPlaceTaskStub(max_result_count=2)
    task.invoke_async(1, {"value": 1})
    task.invoke_async(2, {"value": 2})
    task.invoke_async(3, {"value": 3})  # This should evict request_id=1
    task.invoke_async(4, {"value": 4})  # This should evict request_id=2
    resp1 = task.peek_response(1)
    resp2 = task.peek_response(2)
    resp3 = task.peek_response(3)
    resp4 = task.peek_response(4)
    print("resp1.result_type:", resp1.result_type)
    print("resp2.result_type:", resp2.result_type)
    print("resp3.result_type:", resp3.result_type)
    print("resp4.result_type:", resp4.result_type)
    assert resp1.result_type == PeekResponseResultType.ResponseReceivedButFlushed
    assert resp2.result_type == PeekResponseResultType.ResponseReceivedButFlushed
    assert resp3.result_type == PeekResponseResultType.ResponseFound
    assert resp4.result_type == PeekResponseResultType.ResponseFound


def test_task_threaded_duplicate_request():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        task = ThreadedPoolTaskStub()
        task.initialize(executor=executor)
        assert task.invoke_async(1, {"value": 7})
        assert not task.invoke_async(1, {"value": 8})


def test_task_threaded_max_result_count():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        task = ThreadedPoolTaskStub(max_result_count=2)
        task.initialize(executor=executor)
        task.invoke_async(1, {"value": 1})
        time.sleep(0.15)
        task.invoke_async(2, {"value": 2})
        time.sleep(0.15)
        task.invoke_async(3, {"value": 3})  # This should evict request_id=1
        time.sleep(0.15)
        task.invoke_async(4, {"value": 4})  # This should evict request_id=2
        time.sleep(0.15)
        resp1 = task.peek_response(1)
        resp2 = task.peek_response(2)
        resp3 = task.peek_response(3)
        resp4 = task.peek_response(4)
        print("resp1.result_type:", resp1.result_type)
        print("resp2.result_type:", resp2.result_type)
        print("resp3.result_type:", resp3.result_type)
        print("resp4.result_type:", resp4.result_type)
        assert resp1.result_type == PeekResponseResultType.ResponseReceivedButFlushed
        assert resp2.result_type == PeekResponseResultType.ResponseReceivedButFlushed
        assert resp3.result_type == PeekResponseResultType.ResponseFound
        assert resp4.result_type == PeekResponseResultType.ResponseFound


def test_threaded_result_limit_does_not_cancel_pending_requests():
    release = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(release.wait)
        task = ThreadedPoolTaskStub(max_result_count=1)
        task.initialize(executor=executor)
        assert task.invoke_async(1, {"value": 1})
        assert task.invoke_async(2, {"value": 2})
        assert (
            task.peek_response(1).result_type
            is PeekResponseResultType.RequestSentNoResponse
        )
        assert (
            task.peek_response(2).result_type
            is PeekResponseResultType.RequestSentNoResponse
        )
        release.set()
        blocker.result(timeout=0.5)
        deadline = time.monotonic() + 0.5
        while task.peek_response(2).result_type is not PeekResponseResultType.ResponseFound:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert (
            task.peek_response(1).result_type
            is PeekResponseResultType.ResponseReceivedButFlushed
        )


def test_threaded_submit_failure_releases_request_id_for_retry():
    class FailOnceExecutor:
        def __init__(self, executor):
            self.executor = executor
            self.failed = False

        def submit(self, fn, *args):
            if not self.failed:
                self.failed = True
                raise RuntimeError("submit failed")
            return self.executor.submit(fn, *args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task = ThreadedPoolTaskStub()
        task.initialize(executor=FailOnceExecutor(executor))
        with pytest.raises(RuntimeError, match="submit failed"):
            task.invoke_async(1, {"value": 1})
        assert (
            task.peek_response(1).result_type
            is PeekResponseResultType.ErrorRequestNotSent
        )
        assert task.invoke_async(1, {"value": 1})


def test_task_threaded_cancel():
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task = ThreadedPoolTaskStub(max_result_count=1)
        task.initialize(executor=executor)
        task.invoke_async(1, {"value": 1})
        task.invoke_async(2, {"value": 2})  # This should evict request_id=1
        time.sleep(0.15)
        resp1 = task.peek_response(1)
        print("resp1.result_type:", resp1.result_type)
        assert resp1.result_type in (
            PeekResponseResultType.ErrorUnknown,
            PeekResponseResultType.ErrorRequestNotSent,
            PeekResponseResultType.ResponseFound,
            PeekResponseResultType.ResponseReceivedButFlushed,
        )


def test_threaded_task_rejects_requests_after_stop_until_reinitialized():
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task = ThreadedPoolTaskStub()
        task.initialize(executor=executor)
        task.stop()
        with pytest.raises(RuntimeError, match="initialized"):
            task.invoke_async(1, {"value": 1})
        task.initialize(executor=executor)
        assert task.invoke_async(1, {"value": 1})


if __name__ == "__main__":
    test_task_inplace()
    test_task_threaded()
    test_task_inplace_duplicate_request()
    test_task_inplace_max_result_count()
    test_task_threaded_duplicate_request()
    test_task_threaded_max_result_count()
    test_task_threaded_cancel()
    print("All tests passed.")
