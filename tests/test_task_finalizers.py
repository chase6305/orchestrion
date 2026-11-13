"""SDK payload cleanup may reenter task status through object destructors."""

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

_PILOT_PRUNE_FINALIZER_PROBE = """
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from orchestrion import MoveSyncOption, SimulatedRobotTask, TaskPilot
from orchestrion.tasks import CallableTask, InPlaceFunctionCallTask

armed = threading.Event()
blocked, threads = [], []

class DeviceResult:
    def __deepcopy__(self, memo):
        return DeviceResult()

    def __del__(self):
        if not armed.is_set():
            return
        observed = threading.Event()
        def query():
            pilot.query_health()
            observed.set()
        thread = threading.Thread(target=query)
        threads.append(thread)
        thread.start()
        if not observed.wait(0.25):
            blocked.append(True)

class Task(InPlaceFunctionCallTask):
    def _call_fn(self, request_id, content):
        return {"resource": DeviceResult()}

task = (
    Task() if sys.argv[1] == "inline"
    else CallableTask(lambda request_id, content: {"resource": DeviceResult()})
)
with ThreadPoolExecutor(max_workers=1) as executor, TaskPilot(
    SimulatedRobotTask([0.0]), {"device": task}, executor_owned=executor
) as pilot:
    for _ in range(3):
        request_id = pilot.call_srv_async(
            "device", sync_option=MoveSyncOption.no_sync()
        )
        result = pilot.wait_request(request_id, timeout=1)
        del result
    armed.set()
    try:
        assert pilot.prune_completed_requests(keep_last=0) == 3
    finally:
        armed.clear()
        for thread in threads:
            thread.join(timeout=1)
    assert threads
    assert not any(thread.is_alive() for thread in threads)
    assert not blocked, "SDK finalizer could not observe pilot health during pruning"
print("pilot finalizer completed")
"""


_PILOT_INPUT_FINALIZER_PROBE = """
import sys
import threading

from orchestrion import MoveSyncOption, SimulatedRobotTask, TaskPilot
from orchestrion.tasks import GenericTask, InPlaceFunctionCallTask
from orchestrion.utils.types import PeekResponseResult, PeekResponseResultType

armed = threading.Event()
started = threading.Event()
release = threading.Event()
blocked = []
threads = []

class Input(dict):
    def __deepcopy__(self, memo):
        return type(self)(self)

    def __del__(self):
        if not armed.is_set():
            return
        observed = threading.Event()
        def query():
            pilot.query_health()
            observed.set()
        thread = threading.Thread(target=query)
        threads.append(thread)
        thread.start()
        if not observed.wait(0.25):
            blocked.append(True)

class Pending(GenericTask):
    def invoke_async(self, request_id, content=None):
        return True

    def peek_response(self, request_id):
        return PeekResponseResult(
            PeekResponseResultType.RequestSentNoResponse, request_id
        )

    def cancel_request(self, request_id):
        return True

class Blocking(InPlaceFunctionCallTask):
    def _call_fn(self, request_id, content):
        started.set()
        assert release.wait(2)

with TaskPilot(
    SimulatedRobotTask([0.0]), {'pending': Pending(), 'blocking': Blocking()}
) as pilot:
    request_id = pilot.call_srv_async(
        'pending', Input(value=1), MoveSyncOption.no_sync()
    )
    if sys.argv[1] == 'stop':
        pilot.call_srv_async('pending', Input(value=2), MoveSyncOption.no_sync())
    # Dispatch another call so scheduler snapshots no longer retain the input.
    pilot.call_srv_async('blocking', sync_option=MoveSyncOption.no_sync())
    assert started.wait(1)
    armed.set()
    try:
        if sys.argv[1] == 'cancel':
            assert pilot.cancel_request(request_id)
        else:
            release.set()
            pilot.stop()
    finally:
        armed.clear()
        release.set()
        for thread in threads:
            thread.join(timeout=1)
    assert threads, 'Input was not released'
    assert not any(thread.is_alive() for thread in threads)
    assert not blocked, 'Input finalizer ran with pilot state lock held'
print('pilot input finalizer completed')
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


@pytest.mark.parametrize("backend", ["inline", "threaded"])
def test_pruned_pilot_results_can_observe_health_from_a_cleanup_thread(backend):
    completed = subprocess.run(
        [sys.executable, "-c", _PILOT_PRUNE_FINALIZER_PROBE, backend],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "pilot finalizer completed"


@pytest.mark.parametrize("removal", ["cancel", "stop"])
def test_removed_pilot_input_can_observe_health_from_a_cleanup_thread(removal):
    completed = subprocess.run(
        [sys.executable, "-c", _PILOT_INPUT_FINALIZER_PROBE, removal],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "pilot input finalizer completed"
