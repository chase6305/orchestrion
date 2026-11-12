"""Retries preserve the original command despite mutations by a failed SDK call."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from orchestrion import CallableTask, FieldSpec, RequestSchema, RetryPolicy, TaskFailure


@pytest.mark.parametrize("with_schema", [False, True])
def test_retry_uses_fresh_nested_inputs(with_schema):
    seen = []
    schema = RequestSchema({"values": FieldSpec("array", items=FieldSpec("integer"))})

    def call(request_id, content):
        seen.append(list(content["values"]))
        content["values"].append(99)
        if len(seen) == 1:
            raise ConnectionError("retry")
        return content

    task = CallableTask(
        call, retry_policy=RetryPolicy(max_attempts=2),
        request_schema=schema if with_schema else None,
    )
    original = {"values": [1, 2]}
    with ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0, original)
            task._future_map[0].result(1)
            assert seen == [[1, 2], [1, 2]]
            assert original == {"values": [1, 2]}
            assert task.peek_response(0).content == {"values": [1, 2, 99]}
        finally:
            task.stop()


def test_stop_during_retry_input_copy_prevents_device_call():
    copying, resume = threading.Event(), threading.Event()
    copies = []
    calls = []

    class Payload(dict):
        def __deepcopy__(self, memo):
            copies.append(True)
            if len(copies) == 2:
                copying.set()
                assert resume.wait(2)
            return Payload(self)

    task = CallableTask(
        lambda request_id, content: calls.append(content),
        retry_policy=RetryPolicy(max_attempts=2),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0, Payload(value=1))
            assert copying.wait(1)
            task.stop()
        finally:
            resume.set()
            task.stop()
        outcome = task._future_map[0].result(1)
        assert isinstance(outcome, TaskFailure) and outcome.cancelled
        assert "stopped" in outcome.error
        assert not calls
