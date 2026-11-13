"""TaskPilot synchronization, dispatch, cancellation, and lifecycle races."""

import concurrent.futures
import logging
import threading
import time
import weakref

import pytest

from orchestrion import RequestStatus, SimulatedRobotTask, TaskFailure
from orchestrion.move_sync_option import MoveSyncOption
from orchestrion.task_pilot import TaskPilot
from orchestrion.tasks import CallableTask, GenericTask, InPlaceFunctionCallTask
from orchestrion.tasks.reduced_robot_task_interface import ReducedRobotTaskInterface
from orchestrion.utils.types import PeekResponseResult, PeekResponseResultType

_INVALID_PROGRESS_IDS = [
    pytest.param(False, id="false"),
    pytest.param(True, id="true"),
    pytest.param(0.0, id="float-zero"),
    pytest.param(-1.0, id="float-sentinel"),
    pytest.param(-2, id="below-sentinel"),
]


@pytest.mark.parametrize("synchronized", [False, True])
@pytest.mark.parametrize("rejected", [False, True])
def test_idle_scheduler_releases_completed_request_input(synchronized, rejected):
    reclaimed = threading.Event()
    copies = []

    class TrackedPayload(dict):
        def __deepcopy__(self, memo):
            copied = type(self)(self)
            copies.append(weakref.ref(copied, lambda _: reclaimed.set()))
            return copied

    class Task(InPlaceFunctionCallTask):
        def invoke_async(self, request_id, content=None):
            if rejected:
                return False
            return super().invoke_async(request_id, content)

        def _call_fn(self, request_id, content):
            return None

    with TaskPilot(SimulatedRobotTask([0.0]), {"task": Task()}) as pilot:
        sync = MoveSyncOption.no_sync()
        if synchronized:
            begin, _ = pilot.move_joint_trajectory_async([[1.0]], interval=0.001)
            sync = MoveSyncOption.sync_w_explicit_id(begin)
        request_id = pilot.call_srv_async(
            "task", TrackedPayload(payload="x" * 100000), sync
        )
        result = pilot.wait_request(request_id, timeout=1.0)
        assert result.status is (
            RequestStatus.FAILED if rejected else RequestStatus.SUCCEEDED
        )
        assert reclaimed.wait(1.0)
        assert len(copies) == 1 and copies[0]() is None


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


class SubmoduleRobotTaskStub(RobotTaskStub):
    def query_submodule_state(self, name):
        return self.query_state()


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


@pytest.mark.parametrize(
    "submodule, failure",
    [
        (None, "counter"),
        (None, "history"),
        ("arm", "counter"),
        ("arm", "history"),
        ("arm", "sdk"),
    ],
)
def test_motion_progress_failures_are_shared_within_one_pass_and_recover(
    submodule, failure, caplog
):
    reads, field_reads, formatted = [], [], []

    class DeviceProgressError(RuntimeError):
        def __str__(self):
            formatted.append(True)
            raise ValueError("SDK error formatter failed")

    class InvalidState:
        @property
        def latest_finished_id(self):
            if failure == "counter":
                field_reads.append(True)
                return True
            return 0

        @property
        def cancelled_move_ids(self):
            if failure == "history":
                field_reads.append(True)
                return [[]]
            return ()

    class Robot(SubmoduleRobotTaskStub):
        bad = False

        def query_state(self):
            if self.bad and submodule is None:
                reads.append(True)
                return InvalidState()
            return super().query_state()

        def query_submodule_state(self, name):
            if self.bad and submodule is not None:
                reads.append(True)
                if failure == "sdk":
                    raise DeviceProgressError()
                return InvalidState()
            return super().query_submodule_state(name)

    robot = Robot()
    blocker, synced, working = RecordingTask(), RecordingTask(), RecordingTask()
    option = (
        MoveSyncOption.sync_w_explicit_id(0)
        if submodule is None
        else MoveSyncOption.sync_w_submodule(submodule, 0)
    )
    with TaskPilot(
        robot, {"blocker": blocker, "synced": synced, "working": working}
    ) as pilot:
        try:
            pilot.call_srv_async("blocker", {"block": True}, MoveSyncOption.no_sync())
            assert blocker.started.wait(1)
            ids = [
                pilot.call_srv_async("synced", {"index": i}, option) for i in range(8)
            ]
            robot.bad = True
            immediate = pilot.call_srv_async(
                "working", {"alive": True}, MoveSyncOption.no_sync()
            )
        finally:
            blocker.release.set()
        results = pilot.wait_requests(ids, timeout=1)
        assert all(result.status is RequestStatus.FAILED for result in results)
        assert len({result.error for result in results}) == 1
        assert len(reads) == 1
        assert len(field_reads) == (0 if failure == "sdk" else 1)
        assert len(formatted) == (1 if failure == "sdk" else 0)
        assert synced.calls == []
        assert pilot.wait_request(immediate, timeout=1).content == {"alive": True}
        assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1
        robot.bad = False
        retry = pilot.call_srv_async("synced", {"retry": True}, option)
        assert pilot.wait_request(retry, timeout=1).status is RequestStatus.SUCCEEDED
        assert synced.calls == [{"retry": True}]


def test_repeated_main_progress_errors_log_changes_and_reset_after_recovery(caplog):
    class Robot(RobotTaskStub):
        failure = "controller offline"

        def query_state(self):
            if self.failure is not None:
                raise RuntimeError(self.failure)
            return super().query_state()

    robot = Robot()
    synchronized, working = RecordingTask(), RecordingTask()
    with TaskPilot(
        robot, {"synced": synchronized, "working": working}, poll_interval=5
    ) as pilot:
        request_id = pilot.call_srv_async(
            "synced", sync_option=MoveSyncOption.sync_w_explicit_id(0)
        )

        def next_pass():
            pilot._on_component_change()
            marker = pilot.call_srv_async(
                "working", sync_option=MoveSyncOption.no_sync()
            )
            assert (
                pilot.wait_request(marker, timeout=1).status is RequestStatus.SUCCEEDED
            )

        for _ in range(3):
            next_pass()
        assert pilot.query_request(request_id).status is RequestStatus.WAITING_FOR_MOVE

        def errors():
            return [
                record.getMessage()
                for record in caplog.records
                if "Failed to query robot state" in record.getMessage()
            ]

        assert errors() == ["Failed to query robot state: controller offline"]

        robot.failure = "controller timed out"
        next_pass()
        assert errors()[-1] == "Failed to query robot state: controller timed out"
        assert len(errors()) == 2
        robot.failure = None
        next_pass()
        assert (
            pilot.wait_request(request_id, timeout=1).status is RequestStatus.SUCCEEDED
        )

        robot.failure = "controller offline"
        request_id = pilot.call_srv_async(
            "synced", sync_option=MoveSyncOption.sync_w_explicit_id(0)
        )
        next_pass()
        assert errors()[-1] == "Failed to query robot state: controller offline"
        assert len(errors()) == 3
        robot.failure = None
        next_pass()
        assert (
            pilot.wait_request(request_id, timeout=1).status is RequestStatus.SUCCEEDED
        )
        assert pilot.is_running


@pytest.mark.parametrize(
    "changes",
    [
        {"need_sync": "yes"},
        {"associated_move_id": True},
        {"associated_move_id": -2},
        {"submodule_name": []},
        {"submodule_name": ""},
        {"need_sync": False, "associated_move_id": 0},
        {"need_sync": False, "submodule_name": "arm"},
    ],
)
def test_mutated_sync_options_are_rejected_before_reserving_request_identity(changes):
    task = RecordingTask()
    option = MoveSyncOption.sync_w_latest_move()
    for name, value in changes.items():
        setattr(option, name, value)
    with TaskPilot(SubmoduleRobotTaskStub(), {"device": task}) as pilot:
        assert (
            pilot.call_srv_async("device", sync_option=option, idempotency_key="retry")
            == -1
        )
        assert pilot.timeline() == []
        request_id = pilot.call_srv_async(
            "device", {"ok": True}, MoveSyncOption.no_sync(), idempotency_key="retry"
        )
        assert request_id == 0
        assert pilot.wait_request(request_id, timeout=1).content == {"ok": True}


def test_sync_option_snapshot_is_stable_across_sdk_state_callbacks():
    option = MoveSyncOption.sync_w_submodule("arm", 0)

    class Robot(SubmoduleRobotTaskStub):
        def query_submodule_state(self, name):
            option.associated_move_id = 999
            return super().query_submodule_state(name)

    with TaskPilot(Robot(), {"device": RecordingTask()}) as pilot:
        request_id = pilot.call_srv_async("device", sync_option=option)
        result = pilot.wait_request(request_id, timeout=1)
        assert result.status is RequestStatus.SUCCEEDED
        assert result.associated_move_id == 0
        assert option.associated_move_id == 999


def test_submodule_name_snapshot_uses_plain_string_keys():
    class Name(str):
        __hash__ = None

    with TaskPilot(SubmoduleRobotTaskStub(), {"device": RecordingTask()}) as pilot:
        request_id = pilot.call_srv_async(
            "device", sync_option=MoveSyncOption.sync_w_submodule(Name("arm"), 0)
        )
        result = pilot.wait_request(request_id, timeout=1)
        assert result.status is RequestStatus.SUCCEEDED
        assert result.associated_submodule == "arm"
        assert type(result.associated_submodule) is str


@pytest.mark.parametrize(
    "field", ["priority", "latest_sent", "latest_finished", "explicit"]
)
def test_submission_captures_plain_integer_values_before_queueing(field):
    class Number(int):
        def __int__(self):
            raise RuntimeError("custom integer conversion")

        def __neg__(self):
            raise RuntimeError("custom queue priority")

        def __lt__(self, other):
            raise RuntimeError("custom movement comparison")

        def __le__(self, other):
            raise RuntimeError("custom movement comparison")

    robot = RobotTaskStub()
    priority = Number(5) if field == "priority" else 0
    option = MoveSyncOption.sync_w_latest_move()
    if field == "latest_sent":
        robot.latest_sent_id = Number(10)
    elif field == "latest_finished":
        robot.latest_finished_id = Number(10)
    elif field == "explicit":
        option.associated_move_id = Number(0)
    with TaskPilot(robot, {"device": RecordingTask()}) as pilot:
        request_id = pilot.call_srv_async(
            "device", sync_option=option, priority=priority
        )
        assert request_id == 0
        result = pilot.wait_request(request_id, timeout=1)
        assert result.status is RequestStatus.SUCCEEDED
        assert type(result.priority) is int
        assert type(result.associated_move_id) is int


def test_submission_captures_plain_service_names_and_idempotency_keys():
    class Text(str):
        __hash__ = None

        def __eq__(self, other):
            raise RuntimeError("custom key comparison")

    task = RecordingTask()
    with TaskPilot(RobotTaskStub(), {"device": task}) as pilot:
        request_id = pilot.call_srv_async(
            Text("device"), {"ok": True}, idempotency_key=Text("once")
        )
        assert request_id == 0
        assert pilot.call_srv_async("device", idempotency_key="once") == request_id
        result = pilot.wait_request(request_id, timeout=1)
        assert result.content == {"ok": True}
        assert type(result.service_name) is str
        assert type(result.idempotency_key) is str
        assert task.calls == [{"ok": True}]


@pytest.mark.parametrize("initialization_fails", [False, True])
def test_cleanup_contains_optional_hook_lookup_and_logging_failures(
    initialization_fails, monkeypatch
):
    cleaning = threading.Event()
    admitted = threading.Event()

    class Task(GenericTaskStub):
        @property
        def set_completion_callback(self):
            if cleaning.is_set():
                raise RuntimeError("callback hook unavailable")
            return lambda callback: None

        @property
        def cancel_request(self):
            raise RuntimeError("cancel hook unavailable")

        def stop(self):
            super().stop()
            cleaning.set()

        def invoke_async(self, request_id, content=None):
            result = super().invoke_async(request_id, content)
            admitted.set()
            return result

    class Robot(RobotTaskStub):
        @property
        def set_state_change_callback(self):
            if cleaning.is_set():
                raise RuntimeError("robot callback hook unavailable")
            return lambda callback: None

    class FailingTask(GenericTaskStub):
        def initialize(self, **kwargs):
            raise ValueError("original initialization failure")

    def broken_logger(*args, **kwargs):
        raise RuntimeError("log handler unavailable")

    monkeypatch.setattr("orchestrion.utils.logger.logger.error", broken_logger)
    monkeypatch.setattr("orchestrion.utils.logger.logger.exception", broken_logger)
    robot, first, second = Robot(), Task(), GenericTaskStub()
    services = {"first": first, "second": second}
    if initialization_fails:
        services["broken"] = FailingTask()
    pilot = TaskPilot(robot, services)
    try:
        if initialization_fails:
            with pytest.raises(ValueError, match="original initialization failure"):
                pilot.initialize()
        else:
            pilot.initialize()
            request_id = pilot.call_srv_async("first")
            assert admitted.wait(1)
            pilot.stop()
            assert (
                pilot.wait_request(request_id, timeout=0).status
                is RequestStatus.CANCELLED
            )
        assert first.stopped and second.stopped and robot.stopped
    finally:
        pilot.stop(timeout=1)


@pytest.mark.parametrize("immediate_failure", [False, True])
def test_scheduler_failure_closes_admission_cancels_pending_and_allows_restart(
    immediate_failure, monkeypatch, caplog
):
    release = threading.Event()
    entered = threading.Event()
    formatted = []

    class SchedulerError(RuntimeError):
        def __str__(self):
            formatted.append(True)
            raise ValueError("broken scheduler error formatter")

    robot, task = RobotTaskStub(), RecordingTask()
    pilot = TaskPilot(robot, {"device": task})
    original_scheduler = pilot._run_scheduler

    def fail_scheduler():
        entered.set()
        if not immediate_failure:
            assert release.wait(2)
        raise SchedulerError()

    monkeypatch.setattr(pilot, "_run_scheduler", fail_scheduler)
    try:
        pilot.initialize()
        assert entered.wait(1)
        if not immediate_failure:
            request_id = pilot.call_srv_async("device", {"old": True})
            assert request_id >= 0
        release.set()
        pilot._bg_thread.join(1)
        assert not pilot._bg_thread.is_alive()
        assert not pilot.is_running
        assert not pilot._initialized
        assert robot.stopped
        assert pilot.call_srv_async("device") == -1
        assert formatted == [True]
        assert task.calls == []
        if not immediate_failure:
            result = pilot.wait_request(request_id, timeout=0)
            assert result.status is RequestStatus.CANCELLED
            assert "scheduler failed" in result.error
        assert any("scheduler failed" in record.message for record in caplog.records)

        monkeypatch.setattr(pilot, "_run_scheduler", original_scheduler)
        pilot.initialize()
        fresh = pilot.call_srv_async("device", {"fresh": True})
        assert pilot.wait_request(fresh, timeout=1).content == {"fresh": True}
        assert task.calls == [{"fresh": True}]
    finally:
        release.set()
        pilot.stop(timeout=1)


def test_start_failure_retains_a_launched_scheduler_until_it_exits(monkeypatch):
    release = threading.Event()
    launched = []
    robot, task = RobotTaskStub(), RecordingTask()
    pilot = TaskPilot(robot, {"device": task})
    original_loop = pilot._bg_thread_loop
    original_start = threading.Thread.start

    def parked_scheduler():
        assert release.wait(2)
        original_loop()

    def failed_start(thread):
        original_start(thread)
        launched.append(thread)
        raise RuntimeError("thread started before instrumentation failed")

    monkeypatch.setattr(pilot, "_bg_thread_loop", parked_scheduler)
    try:
        with monkeypatch.context() as failure:
            failure.setattr(threading.Thread, "start", failed_start)
            with pytest.raises(RuntimeError, match="instrumentation failed"):
                pilot.initialize()
        assert len(launched) == 1
        assert pilot._bg_thread is launched[0]
        assert pilot._bg_thread.is_alive()
        assert not pilot.is_running
        assert robot.stopped
        assert pilot.call_srv_async("device") == -1
        with pytest.raises(RuntimeError, match="still stopping"):
            pilot.initialize()
        release.set()
        launched[0].join(1)
        assert not launched[0].is_alive()
        monkeypatch.setattr(pilot, "_bg_thread_loop", original_loop)
        pilot.initialize()
        request_id = pilot.call_srv_async("device", {"fresh": True})
        assert request_id == 0
        assert pilot.wait_request(request_id, timeout=1).content == {"fresh": True}
        assert task.calls == [{"fresh": True}]
    finally:
        release.set()
        pilot.stop(timeout=1)
        for thread in launched:
            thread.join(1)


def test_scheduler_cleanup_attempts_all_devices_when_stop_and_logging_fail(monkeypatch):
    release = threading.Event()
    entered = threading.Event()
    stops = []

    class Task(RecordingTask):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def stop(self):
            stops.append(self.name)
            if self.name == "first":
                raise RuntimeError("device stop failed")

    def broken_logger(*args, **kwargs):
        raise RuntimeError("log handler failed")

    robot = RobotTaskStub()
    pilot = TaskPilot(robot, {"first": Task("first"), "second": Task("second")})

    def fail_scheduler():
        entered.set()
        assert release.wait(2)
        raise RuntimeError("scheduler failed")

    monkeypatch.setattr(pilot, "_run_scheduler", fail_scheduler)
    monkeypatch.setattr("orchestrion.utils.logger.logger.error", broken_logger)
    monkeypatch.setattr("orchestrion.utils.logger.logger.exception", broken_logger)
    try:
        pilot.initialize()
        assert entered.wait(1)
        request_id = pilot.call_srv_async("second")
        release.set()
        pilot._bg_thread.join(1)
        assert not pilot._bg_thread.is_alive()
        assert not pilot.is_running
        assert stops == ["first", "second"]
        assert robot.stopped
        assert (
            pilot.wait_request(request_id, timeout=0).status is RequestStatus.CANCELLED
        )
        assert pilot.call_srv_async("second") == -1
    finally:
        release.set()
        pilot.stop(timeout=1)


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


@pytest.mark.parametrize("submodule_name", [None, "arm"])
def test_motion_batch_shares_history_lookup_and_preserves_priority(submodule_name):
    class CancellationHistory(tuple):
        inspected_ids = 0

        def __iter__(self):
            for move_id in super().__iter__():
                self.inspected_ids += 1
                yield move_id

        def __contains__(self, move_id):
            return any(cancelled_id == move_id for cancelled_id in self)

    history = CancellationHistory((9, 2, 7))

    class Robot(RobotTaskStub):
        def query_state(self):
            return ReducedRobotTaskInterface.ReducedRobotState([], 10, 10, history)

        def query_submodule_state(self, name):
            assert name == "arm"
            return self.query_state()

    task = RecordingTask()
    with TaskPilot(Robot(), {"task": task}, poll_interval=10.0) as pilot:
        blocker = pilot.call_srv_async(
            "task", {"block": True}, MoveSyncOption.no_sync()
        )
        try:
            assert task.started.wait(1.0)
            request_ids = []
            for move_id, priority in ((1, -1), (2, 20), (3, 10), (4, 10)):
                sync_option = (
                    MoveSyncOption.sync_w_explicit_id(move_id)
                    if submodule_name is None
                    else MoveSyncOption.sync_w_submodule(
                        submodule_name, move_id=move_id
                    )
                )
                request_ids.append(
                    pilot.call_srv_async(
                        "task", {"move_id": move_id}, sync_option, priority=priority
                    )
                )
        finally:
            task.release.set()
        results = pilot.wait_requests([blocker] + request_ids, timeout=1.0)
        assert [result.status for result in results] == [
            RequestStatus.SUCCEEDED,
            RequestStatus.SUCCEEDED,
            RequestStatus.CANCELLED,
            RequestStatus.SUCCEEDED,
            RequestStatus.SUCCEEDED,
        ]
        assert task.calls == [
            {"block": True}, {"move_id": 3}, {"move_id": 4}, {"move_id": 1}
        ]
        assert history.inspected_ids == len(history)


def test_motion_waiters_preserve_priority_and_fifo_across_submission_batches():
    class Robot(RobotTaskStub):
        def set_state_change_callback(self, callback):
            self.callback = callback

    robot = Robot()
    robot.latest_finished_id = 4
    task = RecordingTask()
    sync_option = MoveSyncOption.sync_w_explicit_id(5)
    with TaskPilot(robot, {"task": task}, poll_interval=10.0) as pilot:
        cancelled = pilot.call_srv_async(
            "task", {"cancelled": True}, sync_option, priority=50
        )
        first_tie = pilot.call_srv_async(
            "task", {"first": True}, sync_option, priority=10
        )
        assert wait_until(
            lambda: pilot.query_request(first_tie).status
            is RequestStatus.WAITING_FOR_MOVE
        )
        blocker = pilot.call_srv_async(
            "task", {"block": True}, MoveSyncOption.no_sync()
        )
        try:
            assert task.started.wait(1.0)
            high = pilot.call_srv_async(
                "task", {"high": True}, sync_option, priority=20
            )
            second_tie = pilot.call_srv_async(
                "task", {"second": True}, sync_option, priority=10
            )
            assert pilot.cancel_request(cancelled)
            robot.latest_finished_id = 5
            robot.callback()
        finally:
            task.release.set()

        results = pilot.wait_requests(
            [cancelled, first_tie, blocker, high, second_tie], timeout=1.0
        )
        assert [result.status for result in results] == [
            RequestStatus.CANCELLED,
            RequestStatus.SUCCEEDED,
            RequestStatus.SUCCEEDED,
            RequestStatus.SUCCEEDED,
            RequestStatus.SUCCEEDED,
        ]
        assert task.calls == [
            {"block": True},
            {"high": True},
            {"first": True},
            {"second": True},
        ]


def test_unknown_service_is_rejected(caplog):
    pilot = TaskPilot(RobotTaskStub())
    assert pilot.call_srv_async("missing", {}, MoveSyncOption.no_sync()) == -1
    assert "Unknown service" in caplog.text


@pytest.mark.parametrize("threaded", [False, True])
def test_returned_device_cancellation_is_not_a_failed_pilot_request(threaded):
    def call(request_id, content):
        return TaskFailure("controller confirmed stop", cancelled=True)

    class InlineTask(InPlaceFunctionCallTask):
        def _call_fn(self, request_id, content):
            return call(request_id, content)

    task = CallableTask(call) if threaded else InlineTask()
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1) if threaded else None
    )
    with TaskPilot(
        SimulatedRobotTask([0.0]), {"device": task}, executor_owned=executor
    ) as pilot:
        request_id = pilot.call_srv_async(
            "device", sync_option=MoveSyncOption.no_sync()
        )
        result = pilot.wait_request(request_id, timeout=1)
        assert result.status is RequestStatus.CANCELLED
        assert result.error == "controller confirmed stop"
        assert task.peek_response(request_id).cancelled is True
        assert pilot.is_running


def test_preparation_failure_logs_outside_locks_and_preserves_submission_identity(
    monkeypatch,
):
    class BrokenContent(dict):
        def __deepcopy__(self, memo):
            raise RuntimeError("content copy failed")

    messages = []
    with TaskPilot(SimulatedRobotTask([0.0]), {"device": RecordingTask()}) as pilot:

        def log(message, *args):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(lambda: pilot.timeline()).result(1) == []
            messages.append(message % args)

        monkeypatch.setattr("orchestrion.utils.logger.logger.error", log)
        assert (
            pilot.call_srv_async("device", BrokenContent(), idempotency_key="key") == -1
        )
        assert messages == ["Service request was not submitted: content copy failed"]
        request_id = pilot.call_srv_async("device", {"ok": True}, idempotency_key="key")
        assert request_id == 0
        assert pilot.wait_request(request_id, timeout=1).content == {"ok": True}


def test_call_rejects_invalid_sync_option(caplog):
    pilot = TaskPilot(RobotTaskStub(), {"task": GenericTaskStub()})
    pilot.initialize()
    try:
        assert pilot.call_srv_async("task", sync_option=object()) == -1
        assert "sync_option" in caplog.text
    finally:
        pilot.stop()


@pytest.mark.parametrize("submodule_name", [None, "arm"], ids=["main", "submodule"])
@pytest.mark.parametrize("invalid_id", _INVALID_PROGRESS_IDS)
def test_invalid_latest_sent_id_does_not_reserve_a_request_or_idempotency_key(
    submodule_name, invalid_id, caplog
):
    robot = SubmoduleRobotTaskStub()
    robot.latest_sent_id = invalid_id
    robot.latest_finished_id = 0
    task = RecordingTask()
    sync_option = (
        MoveSyncOption.sync_w_latest_move()
        if submodule_name is None
        else MoveSyncOption.sync_w_submodule(submodule_name)
    )
    with TaskPilot(robot, {"device": task}) as pilot:
        assert (
            pilot.call_srv_async(
                "device", {"invalid": True}, sync_option, idempotency_key="retry"
            )
            == -1
        )
        assert "associated_move_id" in caplog.text
        assert task.calls == []
        robot.latest_sent_id = 0
        request_id = pilot.call_srv_async(
            "device", {"valid": True}, sync_option, idempotency_key="retry"
        )
        assert request_id == 0
        result = pilot.wait_request(request_id, timeout=1)
        assert result.status is RequestStatus.SUCCEEDED
        assert result.content == {"valid": True}
        assert task.calls == [{"valid": True}]


@pytest.mark.parametrize("submodule_name", [None, "arm"], ids=["main", "submodule"])
@pytest.mark.parametrize("invalid_id", _INVALID_PROGRESS_IDS)
def test_invalid_latest_finished_id_fails_only_its_synchronized_request(
    submodule_name, invalid_id
):
    robot = SubmoduleRobotTaskStub()
    robot.latest_sent_id = 0
    robot.latest_finished_id = invalid_id
    synchronized, unrelated = RecordingTask(), RecordingTask()
    sync_option = (
        MoveSyncOption.sync_w_explicit_id(0)
        if submodule_name is None
        else MoveSyncOption.sync_w_submodule(submodule_name, move_id=0)
    )
    with TaskPilot(robot, {"synced": synchronized, "unrelated": unrelated}) as pilot:
        request_id = pilot.call_srv_async("synced", sync_option=sync_option)
        result = pilot.wait_request(request_id, timeout=1)
        assert result.status is RequestStatus.FAILED
        assert "latest_finished_id" in result.error
        assert synchronized.calls == []
        next_id = pilot.call_srv_async(
            "unrelated", {"ready": True}, MoveSyncOption.no_sync()
        )
        assert pilot.wait_request(next_id, timeout=1).status is RequestStatus.SUCCEEDED
        assert unrelated.calls == [{"ready": True}]
        assert pilot.is_running


@pytest.mark.parametrize("submodule_name", [None, "arm"], ids=["main", "submodule"])
def test_integer_unfinished_sentinel_keeps_zero_move_pending_until_completion(
    submodule_name,
):
    pending_observed = threading.Event()

    class PendingRobot(SubmoduleRobotTaskStub):
        def query_state(self):
            state = super().query_state()
            if threading.current_thread() is not threading.main_thread():
                pending_observed.set()
            return state

    robot = PendingRobot()
    robot.latest_sent_id = 0
    robot.latest_finished_id = -1
    task = RecordingTask()
    sync_option = (
        MoveSyncOption.sync_w_latest_move()
        if submodule_name is None
        else MoveSyncOption.sync_w_submodule(submodule_name)
    )
    with TaskPilot(robot, {"device": task}) as pilot:
        request_id = pilot.call_srv_async("device", {"ready": True}, sync_option)
        assert pending_observed.wait(1)
        assert task.calls == []
        robot.latest_finished_id = 0
        result = pilot.wait_request(request_id, timeout=1)
        assert result.status is RequestStatus.SUCCEEDED
        assert task.calls == [{"ready": True}]


@pytest.mark.parametrize("submodule_name", [None, "arm"], ids=["main", "submodule"])
@pytest.mark.parametrize("reported_id", [-1, 0], ids=["unset", "zero"])
def test_latest_move_sync_accepts_integer_unset_and_zero_progress(
    submodule_name, reported_id
):
    robot = SubmoduleRobotTaskStub()
    robot.latest_sent_id = robot.latest_finished_id = reported_id
    sync_option = (
        MoveSyncOption.sync_w_latest_move()
        if submodule_name is None
        else MoveSyncOption.sync_w_submodule(submodule_name)
    )
    with TaskPilot(robot, {"device": RecordingTask()}) as pilot:
        request_id = pilot.call_srv_async("device", {"valid": True}, sync_option)
        result = pilot.wait_request(request_id, timeout=1)
        assert result.status is RequestStatus.SUCCEEDED
        assert result.associated_move_id == reported_id


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


def test_request_before_initialization_is_rejected(caplog):
    pilot = TaskPilot(RobotTaskStub(), {"task": GenericTaskStub()})
    assert pilot.call_srv_async("task", {}, MoveSyncOption.no_sync()) == -1
    assert "initialized" in caplog.text


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


def test_pruning_keeps_newest_eligible_results_beside_group_waiters(monkeypatch):
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": RecordingTask()})
    waiting, resume = threading.Event(), threading.Event()
    original_wait = pilot.wait_request

    def paused_wait(request_id, timeout=None):
        if request_id == 0:
            waiting.set()
            assert resume.wait(2)
        return original_wait(request_id, timeout)

    with pilot, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        ids = [pilot.call_srv_async("device") for _ in range(5)]
        pilot.wait_requests(ids, timeout=1)
        monkeypatch.setattr(pilot, "wait_request", paused_wait)
        selected = [ids[0], ids[-1]]
        future = executor.submit(pilot.wait_requests, selected, timeout=1)
        try:
            assert waiting.wait(1)
            assert pilot.prune_completed_requests(keep_last=1) == 2
            for request_id in (ids[1], ids[2]):
                with pytest.raises(KeyError):
                    pilot.query_request(request_id)
            for request_id in (ids[0], ids[3], ids[4]):
                assert pilot.query_request(request_id).status is RequestStatus.SUCCEEDED
        finally:
            resume.set()
        assert [result.request_id for result in future.result(1)] == selected
        assert pilot.prune_completed_requests(keep_last=1) == 2
        assert pilot.query_request(ids[-1]).status is RequestStatus.SUCCEEDED


@pytest.mark.parametrize("stop_before_completion", [False, True])
@pytest.mark.parametrize("failed", [False, True], ids=["success", "sdk-error"])
def test_pruned_cancelled_call_releases_its_response_after_cleanup(
    stop_before_completion, failed
):
    submitted = threading.Event()
    call_started = threading.Event()
    release = threading.Event()
    reclaimed = threading.Event()
    handles = []

    class Payload:
        pass

    def call(request_id, content):
        resource = Payload()
        handles.append(weakref.ref(resource, lambda _: reclaimed.set()))
        call_started.set()
        assert release.wait(2)
        if failed:
            raise RuntimeError("SDK cleanup failed")
        return {"resource": resource}

    class Task(CallableTask):
        def invoke_async(self, request_id, content=None):
            accepted = super().invoke_async(request_id, content)
            submitted.set()
            return accepted

        def cancel_request(self, request_id):
            return True

    task = Task(call)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        with TaskPilot(
            SimulatedRobotTask([0.0]), {"device": task}, executor_owned=executor
        ) as pilot:
            try:
                request_id = pilot.call_srv_async(
                    "device", sync_option=MoveSyncOption.no_sync()
                )
                assert submitted.wait(1)
                assert call_started.wait(1)
                assert pilot.cancel_request(request_id)
                assert pilot.prune_completed_requests(keep_last=0) == 1
                assert not task.forget_response(request_id)
                if stop_before_completion:
                    pilot.stop()
                release.set()
                executor.shutdown(wait=True, cancel_futures=True)
                assert task.peek_status()["retained_results"] == 0
                assert task.peek_status()["pending"] == 0
                assert task.peek_response(request_id).result_type is (
                    PeekResponseResultType.ResponseReceivedButFlushed
                )
                assert reclaimed.wait(1)
                assert handles[0]() is None
            finally:
                release.set()


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
@pytest.mark.parametrize("broken_logger", [False, True])
def test_bulk_cancellation_continues_after_backend_error(
    broken_formatter, broken_logger, caplog, monkeypatch
):
    class CancelError(Exception):
        def __str__(self):
            if broken_formatter:
                raise ValueError("broken formatter")
            return "device cancellation failed"

    admitted = threading.Event()
    attempts = []

    if broken_logger:

        def fail_log(*args, **kwargs):
            raise RuntimeError("log handler unavailable")

        monkeypatch.setattr("orchestrion.utils.logger.logger.error", fail_log)

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
        if not broken_logger:
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
            assert submission.result(1.0) == -1
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
        (
            PeekResponseResult(PeekResponseResultType.ErrorUnknown, error={}),
            "error must be a string",
        ),
        (
            PeekResponseResult(PeekResponseResultType.ErrorUnknown, cancelled="yes"),
            "invalid cancellation flag",
        ),
        (
            PeekResponseResult(PeekResponseResultType.ResponseFound, cancelled=True),
            "invalid cancellation flag",
        ),
        *[
            pytest.param(
                PeekResponseResult(PeekResponseResultType.ResponseFound, request_id),
                "request_id must be an integer",
                id=label,
            )
            for request_id, label in (
                (False, "false"),
                (True, "true"),
                (0.0, "float-zero"),
                (-1.0, "float-sentinel"),
            )
        ],
        pytest.param(
            PeekResponseResult(PeekResponseResultType.ResponseFound, -2),
            "different request",
            id="below-sentinel",
        ),
        pytest.param(
            PeekResponseResult(PeekResponseResultType.ResponseFound, -1),
            None,
            id="unset",
        ),
        pytest.param(
            PeekResponseResult(PeekResponseResultType.ResponseFound, 0),
            None,
            id="zero",
        ),
    ],
)
def test_backend_response_validation_isolates_device_errors(response, error):
    class InconsistentTask(RecordingTask):
        def peek_response(self, request_id):
            return response

    with TaskPilot(
        SimulatedRobotTask([0.0]),
        {"device": InconsistentTask(), "working": RecordingTask()},
    ) as pilot:
        if response.request_id is True:
            # True aliases request 1 just as False aliases request 0.
            primer = pilot.call_srv_async(
                "working", sync_option=MoveSyncOption.no_sync()
            )
            pilot.wait_request(primer, timeout=1)
        request_id = pilot.call_srv_async("device")
        result = pilot.wait_request(request_id, timeout=1.0)
        if error is None:
            assert result.status is RequestStatus.SUCCEEDED
        else:
            assert result.status is RequestStatus.FAILED
            assert error in result.error
        working_id = pilot.call_srv_async(
            "working", sync_option=MoveSyncOption.no_sync()
        )
        assert (
            pilot.wait_request(working_id, timeout=1).status is RequestStatus.SUCCEEDED
        )


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


@pytest.mark.parametrize("broken_logger", [False, True])
def test_inline_initialize_does_not_wait_for_a_caller_joining_the_scheduler(
    broken_logger, monkeypatch
):
    started = threading.Event()
    returned = threading.Event()

    class InitializingTask(RecordingTask):
        def _call_fn(self, request_id, content):
            started.set()
            assert pilot._stop_event.wait(2)
            pilot.initialize()
            returned.set()
            return content

    if broken_logger:
        def fail_log(*args, **kwargs):
            raise RuntimeError("log handler unavailable")

        monkeypatch.setattr("orchestrion.utils.logger.logger.error", fail_log)
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": InitializingTask()})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async("device")
        assert started.wait(1)
        pilot.stop(timeout=1)
        assert returned.is_set()
        assert not pilot._bg_thread.is_alive()
        assert not pilot.is_running
        assert pilot.query_request(request_id).status is RequestStatus.SUCCEEDED
    finally:
        pilot.stop(timeout=1)
