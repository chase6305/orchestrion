"""Result cleanup may reenter task status through SDK object destructors."""

import subprocess
import sys

import pytest

_FINALIZER_PROBE = """
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from orchestrion.tasks import CallableTask, InPlaceFunctionCallTask
from orchestrion.utils.types import PeekResponseResultType

backend, removal = sys.argv[1:]
completed, reclaimed = threading.Event(), threading.Event()

class DeviceResult:
    def __del__(self):
        task.peek_status()
        reclaimed.set()

def call(request_id, content):
    return {"resource": DeviceResult()} if request_id == 0 else None

class InlineTask(InPlaceFunctionCallTask):
    def _call_fn(self, request_id, content):
        return call(request_id, content)

task = (
    CallableTask(call, max_result_count=1)
    if backend == "threaded" else InlineTask(max_result_count=1)
)
task.set_completion_callback(completed.set)
with ThreadPoolExecutor(max_workers=1) as executor:
    task.initialize(executor=executor)
    try:
        assert task.invoke_async(0)
        assert completed.wait(1)
        completed.clear()
        if removal == "forget":
            assert task.forget_response(0)
        else:
            assert task.invoke_async(1)
            assert completed.wait(1)
        assert reclaimed.wait(1)
        assert task.peek_response(0).result_type is (
            PeekResponseResultType.ResponseReceivedButFlushed
        )
        expected_results = 0 if removal == "forget" else 1
        assert task.peek_status()["retained_results"] == expected_results
    finally:
        task.stop()
print("finalizer completed")
"""

_POLLING_FINALIZER_PROBE = """
import threading

from orchestrion.tasks import PollingTask

calls = 0
reclaimed, updated = threading.Event(), threading.Event()

class DeviceSample:
    def __deepcopy__(self, memo):
        return self

    def __del__(self):
        task.peek_status()
        reclaimed.set()

def poll():
    global calls
    calls += 1
    return {"resource": DeviceSample()} if calls == 1 else {"iteration": calls}

def notify():
    if calls >= 2:
        updated.set()

task = PollingTask(poll, interval=0.001)
task.set_completion_callback(notify)
task.initialize()
try:
    assert reclaimed.wait(1)
    assert updated.wait(1)
    assert task.peek_status()["health"] == "online"
    assert task.invoke_async(0)
    assert task.peek_response(0).content["sample"]["iteration"] >= 2
finally:
    task.stop()
print("sample finalizer completed")
"""


@pytest.mark.parametrize("backend", ["inline", "threaded"])
@pytest.mark.parametrize("removal", ["evict", "forget"])
def test_released_results_can_query_status_without_deadlocking(backend, removal):
    # A subprocess timeout also bounds executor shutdown if a regression strands
    # its worker inside a destructor; pytest never inherits that blocked thread.
    completed = subprocess.run(
        [sys.executable, "-c", _FINALIZER_PROBE, backend, removal],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "finalizer completed"


def test_replaced_poll_samples_can_query_status_without_deadlocking():
    completed = subprocess.run(
        [sys.executable, "-c", _POLLING_FINALIZER_PROBE],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "sample finalizer completed"
