"""Function tasks, retries, telemetry, result retention, and lifecycle races."""

import concurrent.futures
import threading
import time
import weakref

import pytest

from orchestrion import RetryPolicy
from orchestrion.tasks.function_call_task import (
    CallableTask,
    InPlaceFunctionCallTask,
    ThreadedPoolFunctionCallTask,
)
from orchestrion.tasks.polling_task import PollingTask
from orchestrion.utils.types import PeekResponseResultType


class InPlaceTaskStub(InPlaceFunctionCallTask):

    def _call_fn(self, request_id, content):
        # Simply return content
        return {"result": content["value"] * 2}


class ThreadedPoolTaskStub(ThreadedPoolFunctionCallTask):
    def _call_fn(self, request_id, content):
        return {"result": content["value"] + 1}


def test_polling_releases_a_rejected_sample_before_the_next_poll():
    observed = threading.Event()
    reclaimed = threading.Event()
    samples = []

    class InvalidSample(list):
        pass

    def poll():
        sample = InvalidSample(["x" * 100000])
        samples.append(weakref.ref(sample, lambda _: reclaimed.set()))
        return sample

    task = PollingTask(poll, interval=10.0)
    task.set_completion_callback(observed.set)
    task.initialize()
    try:
        assert observed.wait(1.0)
        assert "dictionary or None" in task.peek_status()["last_error"]
        assert reclaimed.wait(1.0)
        assert len(samples) == 1 and samples[0]() is None
    finally:
        task.stop()


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
    assert resp.content["result"] == 20


def test_failing_completion_callback_does_not_fail_request():
    task = InPlaceTaskStub()

    def fail_callback():
        raise RuntimeError("callback failure")

    task.set_completion_callback(fail_callback)
    assert task.invoke_async(1, {"value": 10})
    assert task.peek_response(1).result_type is PeekResponseResultType.ResponseFound


def test_completion_callback_must_be_callable_or_none():
    task = InPlaceTaskStub()
    with pytest.raises(TypeError, match="callback"):
        task.set_completion_callback("not callable")
    task.set_completion_callback(None)


def test_task_threaded():
    started, release, completed = threading.Event(), threading.Event(), threading.Event()

    class BlockingTask(ThreadedPoolTaskStub):
        def _call_fn(self, request_id, content):
            started.set()
            assert release.wait(2)
            return super()._call_fn(request_id, content)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task = BlockingTask()
        task.set_completion_callback(completed.set)
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(1, {"value": 5})
            assert started.wait(1)
            assert (
                task.peek_response(1).result_type
                is PeekResponseResultType.RequestSentNoResponse
            )
            release.set()
            assert completed.wait(1)
            assert task.peek_response(1).content == {"result": 6}
            status = task.peek_status()
            assert status["pending"] == 0
            assert status["succeeded"] == 1
            assert status["failed"] == 0
        finally:
            release.set()
            task.stop()


def test_threaded_task_rejects_invalid_executor():
    task = ThreadedPoolTaskStub()
    with pytest.raises(TypeError, match="executor"):
        task.initialize(executor=object())


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


def test_callable_status_does_not_share_nested_device_state():
    device_status = {"device": {"temperatures": [20, 21]}}
    task = CallableTask(lambda request_id, content: content, status=lambda: device_status)
    first = task.peek_status()
    first["device"]["temperatures"].append(99)
    assert device_status == {"device": {"temperatures": [20, 21]}}
    second = task.peek_status()
    device_status["device"]["temperatures"].append(22)
    assert second["device"]["temperatures"] == [20, 21]
    assert task.peek_status()["device"]["temperatures"] == [20, 21, 22]


def test_callable_status_copy_does_not_hold_task_or_retry_lock():
    class Status(dict):
        def __deepcopy__(self, memo):
            for lock in (task._lock, task._retry_lock):
                assert lock.acquire(timeout=0.1)
                lock.release()
            return {"device": {"ready": True}}

    task = CallableTask(lambda request_id, content: content, status=lambda: Status())
    assert task.peek_status()["device"] == {"ready": True}


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


@pytest.mark.parametrize("attempt", [True, 1.5, 0, 1])
def test_retry_policy_rejects_invalid_attempt_number(attempt):
    policy = RetryPolicy(max_attempts=3)
    expected = TypeError if isinstance(attempt, (bool, float)) else ValueError
    with pytest.raises(expected):
        policy.delay_before(attempt)


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


@pytest.fixture(params=[InPlaceTaskStub, ThreadedPoolTaskStub], ids=["inline", "threaded"])
def function_task(request):
    """Share backend setup and wait for retention callbacks before inspecting results."""
    task = request.param(max_result_count=2)
    completed = threading.Event()
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            yield task, completed
        finally:
            task.stop()


def test_function_task_duplicate_request(function_task):
    task, completed = function_task
    assert task.invoke_async(1, {"value": 3})
    assert not task.invoke_async(1, {"value": 4})
    assert completed.wait(1)
    assert task.peek_response(1).result_type is PeekResponseResultType.ResponseFound


def test_function_task_response_is_a_defensive_snapshot(function_task):
    task, completed = function_task
    task._call_fn = lambda request_id, content: {"result": list(content["value"])}
    assert task.invoke_async(1, {"value": [3]})
    assert completed.wait(1)
    response = task.peek_response(1)
    expected = list(response.content["result"])
    response.content["result"].append(99)
    response.content["extra"] = True
    assert task.peek_response(1).content == {"result": expected}


def test_inline_response_copy_can_query_task_status():
    class Content(dict):
        def __deepcopy__(self, memo):
            assert task._lock.acquire(timeout=0.1)
            task._lock.release()
            return {"status": task.peek_status()["retained_results"]}

    class Task(InPlaceFunctionCallTask):
        def _call_fn(self, request_id, content):
            return Content()

    task = Task()
    assert task.invoke_async(1)
    assert task.peek_response(1).content == {"status": 1}


@pytest.mark.parametrize(
    "request_id, expected",
    [(True, TypeError), (1.5, TypeError), (-1, ValueError)],
)
def test_function_tasks_reject_invalid_request_ids(request_id, expected, function_task):
    task, _ = function_task
    with pytest.raises(expected, match="request_id"):
        task.invoke_async(request_id)


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


def test_function_task_max_result_count(function_task):
    task, completed = function_task
    for request_id in (1, 2, 3, 4):
        completed.clear()
        assert task.invoke_async(request_id, {"value": request_id})
        assert completed.wait(1)
    assert [task.peek_response(i).result_type for i in (1, 2, 3, 4)] == [
        PeekResponseResultType.ResponseReceivedButFlushed,
        PeekResponseResultType.ResponseReceivedButFlushed,
        PeekResponseResultType.ResponseFound,
        PeekResponseResultType.ResponseFound,
    ]


def test_pruning_does_not_evict_extra_results_on_later_completion(function_task):
    task, completed = function_task
    for request_id in (10, 20):
        completed.clear()
        assert task.invoke_async(request_id, {"value": request_id})
        assert completed.wait(1)
    assert task.forget_response(20)
    assert not task.forget_response(20)
    for request_id in (5, 30):
        completed.clear()
        assert task.invoke_async(request_id, {"value": request_id})
        assert completed.wait(1)
    for request_id in (10, 30):
        assert task.peek_response(request_id).result_type is PeekResponseResultType.ResponseFound
    for request_id in (5, 20):
        assert task.peek_response(request_id).result_type is PeekResponseResultType.ResponseReceivedButFlushed
        assert not task.invoke_async(request_id, {"value": 1})


def test_threaded_result_limit_does_not_cancel_pending_requests():
    release, completed = threading.Event(), threading.Event()
    notifications = []

    def notify():
        notifications.append(True)
        if len(notifications) == 2:
            completed.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(release.wait, 2)
        task = ThreadedPoolTaskStub(max_result_count=1)
        task.set_completion_callback(notify)
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(1, {"value": 1})
            assert task.invoke_async(2, {"value": 2})
            for request_id in (1, 2):
                assert (
                    task.peek_response(request_id).result_type
                    is PeekResponseResultType.RequestSentNoResponse
                )
            release.set()
            assert blocker.result(timeout=1)
            assert completed.wait(1)
            assert (
                task.peek_response(1).result_type
                is PeekResponseResultType.ResponseReceivedButFlushed
            )
            assert task.peek_response(2).content == {"result": 3}
        finally:
            release.set()
            task.stop()


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


def test_threaded_task_cancels_queued_request_without_dispatch():
    started, release, completed = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def device(request_id, content):
        calls.append(request_id)
        started.set()
        assert release.wait(2)
        return content

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task = CallableTask(device)
        task.set_completion_callback(completed.set)
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0, {"running": True})
            assert started.wait(1)
            assert task.invoke_async(1, {"queued": True})
            assert not task.invoke_async(1, {"duplicate": True})
            assert task.cancel_request(1)
            assert completed.wait(1)
            assert task.peek_response(1).result_type is PeekResponseResultType.ErrorUnknown
            assert task.peek_status()["cancelled"] == 1
            assert task.peek_status()["pending"] == 1
            release.set()
            assert task._future_map[0].result(timeout=1) == {"running": True}
        finally:
            release.set()
            task.stop()
    assert calls == [0]


def test_threaded_task_rejects_requests_after_stop_until_reinitialized():
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task = ThreadedPoolTaskStub()
        task.initialize(executor=executor)
        task.stop()
        with pytest.raises(RuntimeError, match="initialized"):
            task.invoke_async(1, {"value": 1})
        task.initialize(executor=executor)
        assert task.invoke_async(1, {"value": 1})


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("delay", [0.0, 0.01])
def test_stopped_callable_does_not_retry_an_old_invocation(delay, restart):
    started = threading.Event()
    release = threading.Event()
    attempts = []

    def call(request_id, content):
        attempts.append(request_id)
        if request_id == 0:
            started.set()
            assert release.wait(2.0)
            raise ConnectionError("disconnected")
        return content

    task = CallableTask(call, retry_policy=RetryPolicy(max_attempts=3, delay=delay))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        task.initialize(executor=executor)
        task.invoke_async(0)
        try:
            assert started.wait(1.0)
            task.stop()
            if restart:
                task.initialize(executor=executor)
                task.invoke_async(1, {"new": True})
                assert task._future_map[1].result(1.0) == {"new": True}
        finally:
            release.set()
        with pytest.raises(concurrent.futures.CancelledError, match="stopped"):
            task._future_map[0].result(1.0)
        task.stop()
    assert attempts.count(0) == 1


def test_polling_restart_rejects_a_previous_poll_that_is_still_stopping():
    started = threading.Event()
    release = threading.Event()

    def poll():
        started.set()
        assert release.wait(3.0)
        return {"sample": 1}

    task = PollingTask(poll, interval=0.01)
    task.initialize()
    try:
        assert started.wait(1.0)
        task.stop()
        with pytest.raises(RuntimeError, match="still stopping"):
            task.initialize()
    finally:
        release.set()
        task.stop()


def test_polling_concurrent_initialize_starts_one_worker(monkeypatch):
    sample_ready = threading.Event()
    task = PollingTask(lambda: sample_ready.set() or {}, interval=1.0)
    original_start = threading.Thread.start
    worker_starts = []

    def record_start(thread):
        if thread.name == "orchestrion-polling-task":
            worker_starts.append(thread)
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", record_start)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as callers:
            starts = [callers.submit(task.initialize) for _ in range(32)]
            for future in starts:
                future.result(1.0)
        assert sample_ready.wait(1.0)
        assert len(worker_starts) == 1
    finally:
        task.stop()


def test_inline_response_reports_pending_until_call_finishes():
    started = threading.Event()
    release = threading.Event()

    class BlockingTask(InPlaceFunctionCallTask):
        def _call_fn(self, request_id, content):
            started.set()
            assert release.wait(2.0)
            return content

    task = BlockingTask()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task.invoke_async, 0, {"value": 1})
        try:
            assert started.wait(1.0)
            response = task.peek_response(0)
            assert response.result_type is PeekResponseResultType.RequestSentNoResponse
            assert not task.forget_response(0)
        finally:
            release.set()
        assert future.result(1.0)
    assert task.peek_response(0).content == {"value": 1}


def test_inline_failure_is_retained_and_notified():
    class FailingTask(InPlaceFunctionCallTask):
        def _call_fn(self, request_id, content):
            raise ValueError("device rejected command")

    completed = threading.Event()
    task = FailingTask(max_result_count=1)
    task.set_completion_callback(completed.set)
    with pytest.raises(ValueError, match="device rejected"):
        task.invoke_async(0)
    assert completed.is_set()
    response = task.peek_response(0)
    assert response.result_type is PeekResponseResultType.ErrorUnknown
    assert response.error == "device rejected command"
    assert task.forget_response(0)
    assert (
        task.peek_response(0).result_type
        is PeekResponseResultType.ResponseReceivedButFlushed
    )


def test_result_retention_does_not_scan_every_pending_future():
    class CountingFuture(concurrent.futures.Future):
        done_calls = 0

        def done(self):
            self.done_calls += 1
            return super().done()

    class DeferredExecutor:
        def __init__(self):
            self.futures = []

        def submit(self, call, *args):
            future = CountingFuture()
            self.futures.append(future)
            return future

    class EchoTask(ThreadedPoolFunctionCallTask):
        def _call_fn(self, request_id, content):
            return content

    executor = DeferredExecutor()
    task = EchoTask(max_result_count=2)
    task.initialize(executor=executor)
    try:
        for request_id in range(100):
            task.invoke_async(request_id)
        executor.futures[0].set_result({"first": True})
        assert sum(future.done_calls for future in executor.futures) == 0
        for request_id in (2, 1):
            executor.futures[request_id].set_result({"id": request_id})
        assert (
            task.peek_response(0).result_type
            is PeekResponseResultType.ResponseReceivedButFlushed
        )
        assert task.peek_response(1).content == {"id": 1}
        assert task.peek_response(2).content == {"id": 2}
        assert not any(future.cancelled() for future in executor.futures[3:])
    finally:
        task.stop()


def test_late_completion_callback_does_not_restore_pruned_result():
    class DelayedCallbackFuture(concurrent.futures.Future):
        def add_done_callback(self, callback):
            self.callback = callback

    class Executor:
        def __init__(self):
            self.futures = []

        def submit(self, call, *args):
            future = DelayedCallbackFuture()
            future.set_result(call(*args))
            self.futures.append(future)
            return future

    executor = Executor()
    task = ThreadedPoolTaskStub(max_result_count=1)
    task.initialize(executor=executor)
    try:
        assert task.invoke_async(0, {"value": 1})
        assert task.forget_response(0)
        assert task.invoke_async(1, {"value": 2})
        executor.futures[1].callback(executor.futures[1])
        executor.futures[0].callback(executor.futures[0])
        assert task.peek_response(0).result_type is PeekResponseResultType.ResponseReceivedButFlushed
        assert task.peek_response(1).content == {"result": 3}
        assert task.peek_status()["retained_results"] == 1
    finally:
        task.stop()


@pytest.mark.parametrize(
    "policy, attempt, expected",
    [
        (RetryPolicy(delay=0.0, backoff=1e308), 4, 0.0),
        (RetryPolicy(delay=1e-300, backoff=1e200, max_delay=1e200), 4, 1e100),
        (RetryPolicy(delay=2.0, backoff=0.5), 10**400, 0.0),
        (RetryPolicy(delay=2.0, backoff=1.0), 10**400, 2.0),
    ],
)
def test_retry_delay_handles_extreme_exponents_without_changing_the_policy(
    policy, attempt, expected
):
    assert policy.delay_before(attempt) == pytest.approx(expected)


def test_callable_subclass_keeps_its_call_hook():
    observed = []

    class ObservedTask(CallableTask):
        def _call_fn(self, request_id, content):
            observed.append(request_id)
            return super()._call_fn(request_id, content)

    task = ObservedTask(lambda request_id, content: content)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            task.invoke_async(0, {"value": 1})
            assert task._future_map[0].result(timeout=1.0) == {"value": 1}
            assert observed == [0]
        finally:
            task.stop()
