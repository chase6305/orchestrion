"""TaskPilot synchronization, dispatch, cancellation, and lifecycle races."""

import concurrent.futures
import threading
import time

import pytest

from orchestrion import RequestStatus, SimulatedRobotTask
from orchestrion.move_sync_option import MoveSyncOption
from orchestrion.task_pilot import TaskPilot
from orchestrion.tasks import CallableTask, GenericTask, InPlaceFunctionCallTask
from orchestrion.tasks.reduced_robot_task_interface import ReducedRobotTaskInterface
from orchestrion.utils.types import PeekResponseResult, PeekResponseResultType


class RobotTaskStub:
    def __init__(self):
        self.initialized = False
        self.stopped = False
        self.latest_sent_id = 10
        self.latest_finished_id = 10

    def initialize(self):
        self.initialized = True

    def stop(self):
        self.stopped = True

    def query_state(self):
        return ReducedRobotTaskInterface.ReducedRobotState(
            [], self.latest_finished_id, self.latest_sent_id
        )

    def move_joint_trajectory_async(self, **kwargs):
        return 5

    def wait_move(self, time_out=-1, interval=0.05):
        return True


class GenericTaskStub:
    def __init__(self):
        self.initialized = False
        self.stopped = False
        self.invoked = []

    def initialize(self, executor=None):
        self.initialized = True

    def stop(self):
        self.stopped = True

    def invoke_async(self, request_id, content=None):
        self.invoked.append((request_id, content))
        return True


class ClearCallbackFailingRobot(RobotTaskStub):
    def set_state_change_callback(self, callback):
        if callback is None:
            raise RuntimeError("cannot clear robot callback")


class ClearCallbackFailingTask(GenericTaskStub):
    def set_completion_callback(self, callback):
        if callback is None:
            raise RuntimeError("cannot clear task callback")


def wait_until(predicate, timeout=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class RecordingTask(InPlaceFunctionCallTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def _call_fn(self, request_id, content):
        self.calls.append(content)
        if content == {"block": True}:
            self.started.set()
            assert self.release.wait(2.0)
        return content


def test_lifecycle_and_immediate_request():
    robot = RobotTaskStub()
    task = GenericTaskStub()
    pilot = TaskPilot(robot, {"gripper": task})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "gripper", {"close": True}, MoveSyncOption.no_sync()
        )
        assert request_id == 0
        assert wait_until(lambda: task.invoked == [(0, {"close": True})])
    finally:
        pilot.stop()
    assert robot.stopped and task.stopped


def test_stop_continues_when_callback_cleanup_fails():
    robot = ClearCallbackFailingRobot()
    task = ClearCallbackFailingTask()
    pilot = TaskPilot(robot, {"task": task})
    pilot.initialize()
    pilot.stop()
    assert robot.stopped
    assert task.stopped


def test_synchronized_request_waits_for_move():
    robot = RobotTaskStub()
    robot.latest_finished_id = 4
    task = GenericTaskStub()
    pilot = TaskPilot(robot, {"gripper": task})
    pilot.initialize()
    try:
        pilot.call_srv_async(
            "gripper", {}, MoveSyncOption.sync_w_explicit_id(5)
        )
        time.sleep(0.08)
        assert task.invoked == []
        robot.latest_finished_id = 5
        assert wait_until(lambda: len(task.invoked) == 1)
    finally:
        pilot.stop()


def test_unknown_service_is_rejected():
    pilot = TaskPilot(RobotTaskStub())
    with pytest.raises(KeyError):
        pilot.call_srv_async("missing", {}, MoveSyncOption.no_sync())


def test_call_rejects_invalid_sync_option():
    pilot = TaskPilot(RobotTaskStub(), {"task": GenericTaskStub()})
    pilot.initialize()
    try:
        with pytest.raises(TypeError, match="sync_option"):
            pilot.call_srv_async("task", sync_option=object())
    finally:
        pilot.stop()


def test_malformed_submodule_state_fails_request_without_stopping_scheduler():
    class MalformedSubmoduleRobot(RobotTaskStub):
        def query_submodule_state(self, name):
            return object()

    pilot = TaskPilot(MalformedSubmoduleRobot(), {"task": GenericTaskStub()})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "task",
            sync_option=MoveSyncOption.sync_w_submodule("arm", move_id=0),
        )
        result = pilot.wait_request(request_id, timeout=0.5)
        assert result.status.value == "failed"
        assert "cancelled_move_ids" in result.error
        assert pilot.is_running
    finally:
        pilot.stop()


def test_request_before_initialization_is_rejected():
    pilot = TaskPilot(RobotTaskStub(), {"task": GenericTaskStub()})
    with pytest.raises(RuntimeError):
        pilot.call_srv_async("task", {}, MoveSyncOption.no_sync())


@pytest.mark.parametrize("move_id", [True, 1.5, "1"])
def test_explicit_sync_rejects_non_integer_move_ids(move_id):
    with pytest.raises(TypeError, match="integer"):
        MoveSyncOption.sync_w_explicit_id(move_id)


def test_move_sync_option_rejects_invalid_direct_combinations():
    with pytest.raises(TypeError, match="need_sync"):
        MoveSyncOption(need_sync=1)
    with pytest.raises(ValueError, match="move ID"):
        MoveSyncOption(need_sync=False, associated_move_id=0)
    with pytest.raises(ValueError, match="-1"):
        MoveSyncOption(need_sync=True, associated_move_id=-2)
    with pytest.raises(ValueError, match="submodule_name"):
        MoveSyncOption.sync_w_submodule("")
    with pytest.raises(ValueError, match="submodule"):
        MoveSyncOption(need_sync=False, submodule_name="arm")


def test_move_and_wait_are_forwarded():
    pilot = TaskPilot(RobotTaskStub())
    assert pilot.move_joint_trajectory_async([[1, 2, 3]]) == (5, 6)
    assert pilot.wait_move(time_out=0.1)


def test_owned_executor_is_shutdown():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(RobotTaskStub(), {}, executor_owned=executor)
    pilot.initialize()
    pilot.stop()
    with pytest.raises(RuntimeError):
        executor.submit(lambda: None)
    with pytest.raises(RuntimeError, match="cannot restart"):
        pilot.initialize()


def test_pilot_can_restart():
    pilot = TaskPilot(RobotTaskStub())
    pilot.initialize()
    pilot.stop()
    pilot.initialize()
    pilot.stop()


def test_timed_out_scheduler_cannot_be_restarted_until_it_exits():
    task = RecordingTask()
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": task})
    pilot.initialize()
    thread = pilot._bg_thread
    try:
        request_id = pilot.call_srv_async(
            "device", {"block": True}, MoveSyncOption.no_sync()
        )
        assert task.started.wait(1.0)
        with pytest.raises(TimeoutError):
            pilot.stop(timeout=0.0)
        with pytest.raises(RuntimeError, match="still stopping"):
            pilot.initialize()
        assert pilot._bg_thread is thread
        assert pilot.query_request(request_id).status is RequestStatus.CANCELLED
        task.release.set()
        thread.join(1.0)
        assert not thread.is_alive()
        pilot.initialize()
        next_id = pilot.call_srv_async("device", {"next": True})
        assert (
            pilot.wait_request(next_id, timeout=1.0).status is RequestStatus.SUCCEEDED
        )
    finally:
        task.release.set()
        pilot.stop()
        thread.join(1.0)


def test_cancellation_and_pruning_between_selection_and_dispatch(monkeypatch):
    task = RecordingTask()
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": task})
    selected = threading.Event()
    resume = threading.Event()
    original_invoke = pilot._invoke_request

    def paused_invoke(request):
        if request.request_id == 0:
            selected.set()
            assert resume.wait(2.0)
        return original_invoke(request)

    monkeypatch.setattr(pilot, "_invoke_request", paused_invoke)
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "device", sync_option=MoveSyncOption.no_sync()
        )
        assert selected.wait(1.0)
        assert pilot.cancel_request(request_id)
        assert pilot.prune_completed_requests(keep_last=0) == 1
        resume.set()
        next_id = pilot.call_srv_async("device", {"next": True})
        assert (
            pilot.wait_request(next_id, timeout=1.0).status is RequestStatus.SUCCEEDED
        )
        assert task.calls == [{"next": True}]
    finally:
        resume.set()
        pilot.stop()


def test_ready_synchronized_request_competes_with_immediate_priority():
    task = RecordingTask()
    robot = SimulatedRobotTask([0.0])
    pilot = TaskPilot(robot, {"device": task})
    pilot.initialize()
    try:
        blocked = pilot.call_srv_async(
            "device", {"block": True}, MoveSyncOption.no_sync()
        )
        assert task.started.wait(1.0)
        move_id = robot.move_joint_trajectory_async([[1.0]], interval=0.001)
        assert robot.wait_move(time_out=1.0)
        high = pilot.call_srv_async(
            "device",
            {"high": True},
            MoveSyncOption.sync_w_explicit_id(move_id),
            priority=10,
        )
        low = pilot.call_srv_async(
            "device", {"low": True}, MoveSyncOption.no_sync(), priority=-1
        )
        task.release.set()
        pilot.wait_requests([blocked, high, low], timeout=1.0)
        assert task.calls == [{"block": True}, {"high": True}, {"low": True}]
    finally:
        task.release.set()
        pilot.stop()


@pytest.mark.parametrize("response", [None, object(), PeekResponseResult("invalid")])
def test_malformed_task_response_fails_request_without_killing_scheduler(response):
    class BrokenResponseTask(RecordingTask):
        def peek_response(self, request_id):
            return response

    task = RecordingTask()
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]), {"broken": BrokenResponseTask(), "working": task}
    )
    with pilot:
        broken = pilot.call_srv_async("broken")
        result = pilot.wait_request(broken, timeout=1.0)
        assert result.status is RequestStatus.FAILED
        assert "response" in result.error.lower()
        working = pilot.call_srv_async("working")
        assert (
            pilot.wait_request(working, timeout=1.0).status is RequestStatus.SUCCEEDED
        )


def test_group_wait_protects_all_results_from_concurrent_pruning(monkeypatch):
    task = RecordingTask()
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": task})
    waiting = threading.Event()
    resume = threading.Event()
    original_wait = pilot.wait_request

    def paused_wait(request_id, timeout=None):
        if request_id == 0:
            waiting.set()
            assert resume.wait(2.0)
        return original_wait(request_id, timeout)

    with pilot, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        ids = [pilot.call_srv_async("device") for _ in range(2)]
        pilot.wait_requests(ids, timeout=1.0)
        monkeypatch.setattr(pilot, "wait_request", paused_wait)
        future = executor.submit(pilot.wait_requests, ids, timeout=1.0)
        try:
            assert waiting.wait(1.0)
            assert pilot.prune_completed_requests(keep_last=0) == 0
        finally:
            resume.set()
        assert [result.request_id for result in future.result(1.0)] == ids
        assert pilot.prune_completed_requests(keep_last=0) == 2


def test_flushed_backend_response_is_a_terminal_failure():
    class FlushedTask(RecordingTask):
        def peek_response(self, request_id):
            return PeekResponseResult(
                PeekResponseResultType.ResponseReceivedButFlushed,
                request_id=request_id,
            )

    with TaskPilot(SimulatedRobotTask([0.0]), {"device": FlushedTask()}) as pilot:
        request_id = pilot.call_srv_async("device")
        result = pilot.wait_request(request_id, timeout=1.0)
        assert result.status is RequestStatus.FAILED
        assert "flushed" in result.error


def test_stop_closes_dispatch_before_waiting_for_current_callback():
    task = RecordingTask()
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": task})
    pilot.initialize()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        try:
            first = pilot.call_srv_async("device", {"block": True})
            assert task.started.wait(1.0)
            queued = pilot.call_srv_async("device", {"queued": True})
            stopping = executor.submit(pilot.stop, timeout=1.0)
            assert pilot._stop_event.wait(1.0)
            task.release.set()
            stopping.result(1.0)
            assert task.calls == [{"block": True}]
            assert pilot.query_request(queued).status is RequestStatus.CANCELLED
            assert pilot.query_request(first).status.terminal
            assert pilot._task_queue.empty()
        finally:
            task.release.set()
            pilot.stop()


def test_group_wait_validates_every_id_before_waiting_and_releases_pins():
    with TaskPilot(SimulatedRobotTask([0.0]), {"device": RecordingTask()}) as pilot:
        waiting = pilot.call_srv_async(
            "device", sync_option=MoveSyncOption.sync_w_explicit_id(99)
        )
        with pytest.raises(KeyError, match="Unknown request"):
            pilot.wait_requests([waiting, 999], timeout=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            pilot.wait_requests([waiting, -1], timeout=0.0)
        with pytest.raises(TimeoutError):
            pilot.wait_requests([waiting, waiting], timeout=0.0)
        assert pilot.cancel_request(waiting)
        results = pilot.wait_requests([waiting, waiting], timeout=0.0)
        assert len(results) == 2
        assert pilot.prune_completed_requests(keep_last=0) == 1


@pytest.mark.parametrize("accepted", [True, False])
def test_backend_cancellation_can_notify_from_another_thread(accepted):
    class CancellableTask(RecordingTask):
        def peek_response(self, request_id):
            return PeekResponseResult(
                PeekResponseResultType.RequestSentNoResponse, request_id=request_id
            )

        def cancel_request(self, request_id):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                notification = executor.submit(self._notify_completion)
                # A backend may await its worker acknowledging cancellation.
                # It must be able to notify the pilot without waiting on its state lock.
                notification.result(timeout=0.5)
            finally:
                executor.shutdown(wait=False)
            return accepted

    task = CancellableTask()
    with TaskPilot(SimulatedRobotTask([0.0]), {"device": task}) as pilot:
        request_id = pilot.call_srv_async("device", {"block": True})
        try:
            assert task.started.wait(1.0)
            assert pilot.cancel_request(request_id) is accepted
        finally:
            task.release.set()
        expected = RequestStatus.CANCELLED if accepted else RequestStatus.RUNNING
        assert pilot.query_request(request_id).status is expected


@pytest.mark.parametrize("accepted", [True, False, None])
def test_completion_during_cancellation_preserves_the_final_outcome(accepted):
    entered_cancel = threading.Event()
    finish_cancel = threading.Event()
    completed = threading.Event()

    class CancellableTask(RecordingTask):
        def cancel_request(self, request_id):
            entered_cancel.set()
            assert finish_cancel.wait(2.0)
            if accepted is None:
                raise RuntimeError("cancellation transport failed")
            return accepted

        def peek_response(self, request_id):
            response = super().peek_response(request_id)
            if response.result_type is PeekResponseResultType.ResponseFound:
                completed.set()
            return response

    task = CancellableTask()
    with TaskPilot(SimulatedRobotTask([0.0]), {"device": task}) as pilot:
        request_id = pilot.call_srv_async("device", {"block": True})
        assert task.started.wait(1.0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            cancel = executor.submit(pilot.cancel_request, request_id)
            try:
                assert entered_cancel.wait(1.0)
                task.release.set()
                assert completed.wait(1.0)
                assert pilot.query_request(request_id).status is RequestStatus.RUNNING
            finally:
                task.release.set()
                finish_cancel.set()
            if accepted is None:
                with pytest.raises(RuntimeError, match="cancellation transport failed"):
                    cancel.result(1.0)
            else:
                assert cancel.result(1.0) is accepted
        expected = RequestStatus.CANCELLED if accepted else RequestStatus.SUCCEEDED
        assert pilot.wait_request(request_id, timeout=1.0).status is expected
        assert not pilot._cancelling_requests


@pytest.mark.parametrize("broken_formatter", [False, True])
def test_bulk_cancellation_continues_after_backend_error(broken_formatter, caplog):
    class CancelError(Exception):
        def __str__(self):
            if broken_formatter:
                raise ValueError("broken formatter")
            return "device cancellation failed"

    admitted = threading.Event()
    attempts = []

    class Task(GenericTaskStub):
        def invoke_async(self, request_id, content=None):
            super().invoke_async(request_id, content)
            if len(self.invoked) == 3:
                admitted.set()
            return True

        def cancel_request(self, request_id):
            attempts.append(request_id)
            if request_id == 0:
                raise CancelError()
            return True

    with TaskPilot(SimulatedRobotTask([0.0]), {"device": Task()}) as pilot:
        ids = [pilot.call_srv_async("device") for _ in range(3)]
        assert admitted.wait(1)
        assert pilot.cancel_all_requests() == ids[1:]
        assert attempts == ids
        assert pilot.query_request(ids[0]).status is RequestStatus.RUNNING
        for request_id in ids[1:]:
            assert pilot.query_request(request_id).status is RequestStatus.CANCELLED
        assert not pilot._cancelling_requests
        assert "Failed to cancel request 0" in caplog.text


def test_request_prepared_before_stop_cannot_enter_a_restarted_pilot():
    copying = threading.Event()
    resume = threading.Event()

    class PausedContent(dict):
        def __deepcopy__(self, memo):
            copying.set()
            assert resume.wait(2.0)
            return {"stale": True}

    task = RecordingTask()
    with TaskPilot(SimulatedRobotTask([0.0]), {"device": task}) as pilot:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            submission = executor.submit(
                pilot.call_srv_async, "device", PausedContent()
            )
            try:
                assert copying.wait(1.0)
                pilot.stop()
                pilot.initialize()
            finally:
                resume.set()
            with pytest.raises(RuntimeError, match="stopped"):
                submission.result(1.0)
        assert task.calls == []
        request_id = pilot.call_srv_async("device", {"fresh": True})
        assert pilot.wait_request(request_id, timeout=1.0).content == {"fresh": True}


@pytest.mark.parametrize(
    "response, error",
    [
        (
            PeekResponseResult(PeekResponseResultType.ResponseFound, 999),
            "different request",
        ),
        (PeekResponseResult(PeekResponseResultType.ErrorRequestNotSent), "no record"),
    ],
)
def test_inconsistent_backend_response_fails_the_affected_request(response, error):
    class InconsistentTask(RecordingTask):
        def peek_response(self, request_id):
            return response

    with TaskPilot(SimulatedRobotTask([0.0]), {"device": InconsistentTask()}) as pilot:
        request_id = pilot.call_srv_async("device")
        result = pilot.wait_request(request_id, timeout=1.0)
        assert result.status is RequestStatus.FAILED
        assert error in result.error


def test_backend_attribute_error_is_reported_instead_of_waiting_forever():
    class BrokenTask(RecordingTask):
        def peek_response(self, request_id):
            raise AttributeError("device response decoder is missing")

    with TaskPilot(SimulatedRobotTask([0.0]), {"device": BrokenTask()}) as pilot:
        request_id = pilot.call_srv_async("device")
        result = pilot.wait_request(request_id, timeout=1.0)
        assert result.status is RequestStatus.FAILED
        assert "decoder" in result.error


@pytest.mark.parametrize("failure_at", ["invoke", "response", "copy"])
def test_broken_sdk_exception_does_not_stop_dispatch(failure_at):
    class BrokenError(Exception):
        def __str__(self):
            raise ValueError("broken SDK formatter")

    class BrokenContent(dict):
        def __deepcopy__(self, memo):
            raise BrokenError()

    class BrokenTask(RecordingTask):
        def invoke_async(self, request_id, content=None):
            if failure_at == "invoke":
                raise BrokenError()
            return True

        def peek_response(self, request_id):
            if failure_at == "response":
                raise BrokenError()
            return PeekResponseResult(
                PeekResponseResultType.ResponseFound, request_id, BrokenContent()
            )

    with TaskPilot(
        SimulatedRobotTask([0.0]),
        {"broken": BrokenTask(), "working": RecordingTask()},
    ) as pilot:
        broken = pilot.call_srv_async("broken")
        result = pilot.wait_request(broken, timeout=1)
        assert result.status is RequestStatus.FAILED
        assert "BrokenError" in result.error
        working = pilot.call_srv_async("working", {"ok": True})
        assert pilot.wait_request(working, timeout=1).status is RequestStatus.SUCCEEDED


def test_health_isolates_broken_sdk_exception_formatters():
    class BrokenError(Exception):
        def __str__(self):
            raise ValueError("broken SDK formatter")

    class BrokenRobot(RobotTaskStub):
        def query_state(self):
            raise BrokenError()

    class BrokenTask(RecordingTask):
        def peek_status(self):
            raise BrokenError()

    pilot = TaskPilot(
        BrokenRobot(), {"broken": BrokenTask(), "working": RecordingTask()}
    )
    statuses = pilot.query_all_task_statuses()
    assert not statuses["broken"]["available"]
    assert "BrokenError" in statuses["broken"]["error"]
    assert "working" in statuses
    health = pilot.query_health()
    assert not health["robot"]["available"]
    assert "BrokenError" in health["robot"]["error"]


@pytest.mark.parametrize("reader", ["query_request", "wait_request"])
def test_slow_result_copy_does_not_block_other_requests(reader):
    armed, copying, release = threading.Event(), threading.Event(), threading.Event()

    class SlowContent(dict):
        def __deepcopy__(self, memo):
            if armed.is_set():
                copying.set()
                assert release.wait(2)
            return SlowContent(self)

    class SlowTask(RecordingTask):
        def _call_fn(self, request_id, content):
            return SlowContent(value=1)

    with TaskPilot(
        SimulatedRobotTask([0.0]),
        {"slow": SlowTask(), "working": RecordingTask()},
    ) as pilot, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        slow = pilot.call_srv_async("slow")
        assert pilot.wait_request(slow, timeout=1).content == {"value": 1}
        armed.set()
        future = executor.submit(getattr(pilot, reader), slow)
        try:
            assert copying.wait(1)
            working = pilot.call_srv_async("working", {"ok": True})
            assert pilot.wait_request(working, timeout=1).status is RequestStatus.SUCCEEDED
            if reader == "wait_request":
                assert pilot.prune_completed_requests(keep_last=0) == 1
        finally:
            release.set()
        assert future.result(1).content == {"value": 1}
        assert not pilot._request_waiters


def test_failed_result_copy_releases_waiter_and_preserves_original_error():
    armed = threading.Event()

    class Uncopyable(dict):
        def __deepcopy__(self, memo):
            if armed.is_set():
                raise KeyError("payload copy failed")
            return Uncopyable(self)

    class Task(RecordingTask):
        def _call_fn(self, request_id, content):
            return Uncopyable(value=1)

    with TaskPilot(SimulatedRobotTask([0.0]), {"device": Task()}) as pilot:
        request_id = pilot.call_srv_async("device")
        pilot.wait_request(request_id, timeout=1)
        armed.set()
        for reader in (pilot.query_request, pilot.wait_request):
            with pytest.raises(KeyError, match="payload copy failed"):
                reader(request_id)
        assert not pilot._request_waiters
        assert pilot.prune_completed_requests(keep_last=0) == 1


@pytest.mark.parametrize("wait_kind", ["request", "group", "health"])
def test_scheduler_rejects_reentrant_blocking_waits(wait_kind):
    class WaitingTask(RecordingTask):
        def _call_fn(self, request_id, content):
            if wait_kind == "request":
                pilot.wait_request(request_id, timeout=0.1)
            elif wait_kind == "group":
                pilot.wait_requests([request_id, request_id], timeout=0.1)
            else:
                pilot.wait_health_change(pilot.health_revision, timeout=0.1)
            return {}

    with TaskPilot(
        SimulatedRobotTask([0.0]),
        {"waiting": WaitingTask(), "working": RecordingTask()},
    ) as pilot:
        request_id = pilot.call_srv_async("waiting")
        result = pilot.wait_request(request_id, timeout=1)
        assert result.status is RequestStatus.FAILED
        assert "scheduler callbacks cannot block" in result.error
        assert not pilot._request_waiters
        working = pilot.call_srv_async("working", {"ok": True})
        assert pilot.wait_request(working, timeout=1).status is RequestStatus.SUCCEEDED


def test_scheduler_can_read_completed_results_and_use_zero_timeout_waits():
    class InspectingTask(RecordingTask):
        def _call_fn(self, request_id, content):
            assert pilot.wait_request(completed).status is RequestStatus.SUCCEEDED
            assert len(pilot.wait_requests([completed, completed])) == 2
            assert pilot.wait_requests([]) == []
            with pytest.raises(TimeoutError):
                pilot.wait_request(request_id, timeout=0)
            assert pilot.wait_health_change(pilot.health_revision, timeout=0) is None
            return {"ok": True}

    with TaskPilot(
        SimulatedRobotTask([0.0]),
        {"inspect": InspectingTask(), "working": RecordingTask()},
    ) as pilot:
        completed = pilot.call_srv_async("working")
        pilot.wait_request(completed, timeout=1)
        inspected = pilot.call_srv_async("inspect")
        assert pilot.wait_request(inspected, timeout=1).content == {"ok": True}
        assert not pilot._request_waiters


@pytest.mark.parametrize("fail_initialization", [False, True])
def test_shared_task_lifecycle_runs_once_per_instance(fail_initialization):
    class LifecycleTask(RecordingTask):
        def __init__(self):
            super().__init__()
            self.initializations = 0
            self.stops = 0
            self.callback_changes = []

        def initialize(self, **kwargs):
            self.initializations += 1
            assert self.initializations == 1

        def stop(self):
            self.stops += 1

        def set_completion_callback(self, callback):
            self.callback_changes.append(callback)
            super().set_completion_callback(callback)

    class FailingTask(RecordingTask):
        def initialize(self, **kwargs):
            raise RuntimeError("device unavailable")

    shared = LifecycleTask()
    services = {"first": shared, "second": shared}
    if fail_initialization:
        services["broken"] = FailingTask()
    pilot = TaskPilot(SimulatedRobotTask([0.0]), services)
    try:
        if fail_initialization:
            with pytest.raises(RuntimeError, match="device unavailable"):
                pilot.initialize()
        else:
            pilot.initialize()
            for alias in services:
                request_id = pilot.call_srv_async(alias)
                assert pilot.wait_request(request_id, timeout=1).status is RequestStatus.SUCCEEDED
    finally:
        pilot.stop()
    assert shared.initializations == 1
    assert shared.stops == 1
    assert len(shared.callback_changes) == 2
    assert callable(shared.callback_changes[0])
    assert shared.callback_changes[1] is None


def test_pending_backlog_does_not_poll_every_request_after_each_submission():
    count = 100
    admitted = threading.Event()

    class PendingTask(GenericTask):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.polls = 0

        def invoke_async(self, request_id, content=None):
            self.calls += 1
            if self.calls == count:
                admitted.set()
            return True

        def peek_response(self, request_id):
            self.polls += 1
            return PeekResponseResult(
                PeekResponseResultType.RequestSentNoResponse, request_id=request_id
            )

    task = PendingTask()
    with TaskPilot(
        SimulatedRobotTask([0.0]), {"device": task}, poll_interval=5.0
    ) as pilot:
        for _ in range(count):
            pilot.call_srv_async("device", sync_option=MoveSyncOption.no_sync())
        assert admitted.wait(1.0)
        assert task.polls <= count * 2


def test_shared_task_aliases_all_receive_completion_notifications():
    release = threading.Event()
    started = threading.Event()

    def call(request_id, content):
        started.set()
        assert release.wait(2.0)
        return content

    task = CallableTask(call)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    with TaskPilot(
        SimulatedRobotTask([0.0]),
        {"first_alias": task, "second_alias": task},
        executor_owned=executor,
        poll_interval=5.0,
    ) as pilot:
        try:
            request_id = pilot.call_srv_async("first_alias", {"value": 1})
            assert started.wait(1.0)
            release.set()
            result = pilot.wait_request(request_id, timeout=1.0)
            assert result.status is RequestStatus.SUCCEEDED
        finally:
            release.set()


def test_inline_task_can_stop_its_pilot_without_joining_itself():
    stopped = threading.Event()

    class StoppingTask(RecordingTask):
        def _call_fn(self, request_id, content):
            pilot.stop()
            return content

        def stop(self):
            stopped.set()

    task = StoppingTask()
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": task})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async("device")
        assert stopped.wait(1.0)
        result = pilot.wait_request(request_id, timeout=1.0)
        assert result.status is RequestStatus.CANCELLED
    finally:
        pilot.stop(timeout=1.0)
    assert not pilot._bg_thread.is_alive()


@pytest.mark.parametrize("metadata", [{"bad": {1, 2}}, {"bad": float("nan")}])
def test_non_json_device_status_cannot_break_the_health_endpoint(metadata):
    class StatusTask(RecordingTask):
        def peek_status(self):
            return metadata

    with TaskPilot(
        SimulatedRobotTask([0.0]), {"broken": StatusTask(), "working": RecordingTask()}
    ) as pilot:
        health = pilot.query_health()
        assert health["peripherals"]["working"]["available"]
        assert not health["peripherals"]["broken"]["available"]
        assert "JSON-compatible" in health["peripherals"]["broken"]["error"]
        assert not health["healthy"]


def test_device_availability_must_be_boolean():
    class StatusTask(RecordingTask):
        def peek_status(self):
            return {"available": "false"}

    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": StatusTask()})
    with pytest.raises(ValueError, match="boolean"):
        pilot.query_task_status("device")


def test_inline_stop_does_not_wait_for_an_external_caller_joining_the_scheduler():
    started = threading.Event()
    returned = threading.Event()

    class StoppingTask(RecordingTask):
        def _call_fn(self, request_id, content):
            started.set()
            assert pilot._stop_event.wait(2.0)
            pilot.stop()
            returned.set()
            return content

    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": StoppingTask()})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async("device")
        assert started.wait(1.0)
        pilot.stop(timeout=1.0)
        assert returned.is_set()
        assert pilot.query_request(request_id).status is RequestStatus.SUCCEEDED
    finally:
        pilot.stop()
