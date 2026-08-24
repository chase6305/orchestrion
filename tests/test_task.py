import concurrent.futures
import time

import pytest

from orchestrion.tasks.function_call_task import (
    InPlaceFunctionCallTask,
    ThreadedPoolFunctionCallTask,
)
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


def test_task_inplace_duplicate_request():
    task = InPlaceTaskStub()
    assert task.invoke_async(1, {"value": 3})
    # Duplicate request_id should return False
    assert not task.invoke_async(1, {"value": 4})


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
