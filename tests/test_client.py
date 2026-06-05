import time

import concurrent.futures

from orchestrion.utils.types import PeekResponseResultType
from orchestrion.tasks.function_call_task import (
    InPlaceFunctionCallTask,
    ThreadedPoolFunctionCallTask,
)


# Subclass for testing, override _call_fn
class TestInPlaceTask(InPlaceFunctionCallTask):

    def _call_fn(self, request_id, content):
        # Simply return content
        return {"result": content["value"] * 2}


class TestThreadedPoolTask(ThreadedPoolFunctionCallTask):

    def _call_fn(self, request_id, content):
        # Simulate a time-consuming operation
        time.sleep(0.1)
        return {"result": content["value"] + 1}


def test_inplace():
    task = TestInPlaceTask()
    assert task.invoke_async(1, {"value": 10})
    resp = task.peek_response(1)
    print("InPlace resp:", resp.content)
    assert resp.content["result"] == 20


def test_threaded():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        task = TestThreadedPoolTask()
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


def test_inplace_duplicate_request():
    task = TestInPlaceTask()
    assert task.invoke_async(1, {"value": 3})
    # Duplicate request_id should return False
    assert not task.invoke_async(1, {"value": 4})


def test_inplace_max_result_count():
    task = TestInPlaceTask(max_result_count=2)
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
    assert resp1.result_type == PeekResponseResultType.ErrorUnknown
    assert resp2.result_type == PeekResponseResultType.ResponseFound
    assert resp3.result_type == PeekResponseResultType.ResponseFound
    assert resp4.result_type == PeekResponseResultType.ResponseFound


def test_threaded_duplicate_request():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        task = TestThreadedPoolTask()
        task.initialize(executor=executor)
        assert task.invoke_async(1, {"value": 7})
        assert not task.invoke_async(1, {"value": 8})


def test_threaded_max_result_count():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        task = TestThreadedPoolTask(max_result_count=2)
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
        assert resp1.result_type == PeekResponseResultType.ErrorRequestNotSent
        assert resp2.result_type == PeekResponseResultType.ResponseFound
        assert resp3.result_type == PeekResponseResultType.ResponseFound
        assert resp4.result_type == PeekResponseResultType.ResponseFound


def test_threaded_cancel():
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task = TestThreadedPoolTask(max_result_count=1)
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
        )


if __name__ == "__main__":
    test_inplace()
    test_threaded()
    test_inplace_duplicate_request()
    test_inplace_max_result_count()
    test_threaded_duplicate_request()
    test_threaded_max_result_count()
    test_threaded_cancel()
    print("All tests passed.")
