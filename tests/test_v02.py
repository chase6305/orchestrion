import concurrent.futures
import json
import threading
import time

import pytest

from orchestrion import (
    MoveSyncOption,
    RequestStatus,
    SimulatedRobotTask,
    TaskPilot,
)
from orchestrion.tasks.function_call_task import (
    InPlaceFunctionCallTask,
    ThreadedPoolFunctionCallTask,
)


class EchoTask(InPlaceFunctionCallTask):
    def _call_fn(self, request_id, content):
        return {"request_id": request_id, **(content or {})}


class CountingEchoTask(EchoTask):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.call_lock = threading.Lock()

    def _call_fn(self, request_id, content):
        with self.call_lock:
            self.call_count += 1
        return super()._call_fn(request_id, content)


class FailingTask(InPlaceFunctionCallTask):
    def _call_fn(self, request_id, content):
        raise ValueError("broken peripheral")


class StatusFailingTask(EchoTask):
    def peek_status(self):
        raise ConnectionError("device offline")


class SlowTask(ThreadedPoolFunctionCallTask):
    def _call_fn(self, request_id, content):
        return content


class DelayedTask(ThreadedPoolFunctionCallTask):
    def _call_fn(self, request_id, content):
        time.sleep(0.02)
        return content


class AsyncFailingTask(ThreadedPoolFunctionCallTask):
    def _call_fn(self, request_id, content):
        raise RuntimeError("async failure")


class BlockingTask(ThreadedPoolFunctionCallTask):
    def __init__(self, release):
        super().__init__()
        self.release = release
        self.started = threading.Event()

    def _call_fn(self, request_id, content):
        self.started.set()
        self.release.wait()
        return content


class OrderedBlockingTask(InPlaceFunctionCallTask):
    def __init__(self):
        super().__init__()
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.execution_order = []

    def _call_fn(self, request_id, content):
        label = content["label"]
        self.execution_order.append(label)
        if label == "first":
            self.first_started.set()
            self.release_first.wait()
        return {"label": label}


class PartiallyFailingRobot(SimulatedRobotTask):
    def initialize(self):
        super().initialize()
        raise RuntimeError("robot initialization failed")


class CallbackFailingRobot(SimulatedRobotTask):
    def set_state_change_callback(self, callback):
        if callback is not None:
            raise RuntimeError("callback registration failed")
        super().set_state_change_callback(callback)


class CountingRobot(SimulatedRobotTask):
    def __init__(self):
        super().__init__([0.0])
        self.initialize_count = 0
        self.stop_count = 0
        self.count_lock = threading.Lock()

    def initialize(self):
        with self.count_lock:
            self.initialize_count += 1
        time.sleep(0.02)
        return super().initialize()

    def stop(self):
        with self.count_lock:
            self.stop_count += 1
        return super().stop()


class LifecycleTask(EchoTask):
    def __init__(self, fail_initialize=False):
        super().__init__()
        self.fail_initialize = fail_initialize
        self.stopped = False

    def initialize(self, **kwargs):
        if self.fail_initialize:
            raise RuntimeError("initialization failed")

    def stop(self):
        self.stopped = True


def test_request_success_wait_and_timeline():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "echo", {"value": 3}, MoveSyncOption.no_sync()
        )
        result = pilot.wait_request(request_id, timeout=0.5)
        assert result.status is RequestStatus.SUCCEEDED
        assert result.content == {"request_id": request_id, "value": 3}
        assert result.started_at is not None
        assert result.finished_at >= result.started_at
        assert [event.status for event in pilot.timeline(request_id)] == [
            RequestStatus.QUEUED,
            RequestStatus.RUNNING,
            RequestStatus.SUCCEEDED,
        ]
        assert pilot.query_task_status("echo") == {
            "available": True,
            "execution": "in_place",
            "health": "online",
            "latest_request_id": request_id,
            "observed_at": pytest.approx(time.time(), abs=1.0),
            "retained_results": 1,
        }
        assert pilot.query_all_task_statuses()["echo"]["retained_results"] == 1
    finally:
        pilot.stop()


def test_wait_requests_preserves_order_and_uses_shared_deadline():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]),
        {"slow": DelayedTask()},
        executor_owned=executor,
    )
    pilot.initialize()
    try:
        request_ids = [
            pilot.call_srv_async(
                "slow", {"value": value}, MoveSyncOption.no_sync()
            )
            for value in (1, 2)
        ]
        results = pilot.wait_requests(request_ids, timeout=0.5)
        assert [result.request_id for result in results] == request_ids
        assert [result.content["value"] for result in results] == [1, 2]
        with pytest.raises(ValueError, match="timeout"):
            pilot.wait_requests([], timeout=True)
        with pytest.raises(TypeError, match="request_ids"):
            pilot.wait_requests([True])
    finally:
        pilot.stop()


def test_request_priority_reorders_queued_work_and_preserves_fifo_ties():
    task = OrderedBlockingTask()
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": task})
    pilot.initialize()
    try:
        first = pilot.call_srv_async(
            "device", {"label": "first"}, MoveSyncOption.no_sync()
        )
        assert task.first_started.wait(0.5)
        low = pilot.call_srv_async(
            "device", {"label": "low"}, MoveSyncOption.no_sync(), priority=-5
        )
        high_a = pilot.call_srv_async(
            "device", {"label": "high-a"}, MoveSyncOption.no_sync(), priority=10
        )
        high_b = pilot.call_srv_async(
            "device", {"label": "high-b"}, MoveSyncOption.no_sync(), priority=10
        )
        task.release_first.set()
        pilot.wait_requests([first, low, high_a, high_b], timeout=0.5)
        assert task.execution_order == ["first", "high-a", "high-b", "low"]
        assert pilot.query_request(high_a).priority == 10
        with pytest.raises(TypeError, match="priority"):
            pilot.call_srv_async("device", priority=True)
    finally:
        task.release_first.set()
        pilot.stop()


def test_idempotency_key_deduplicates_concurrent_service_calls():
    task = CountingEchoTask()
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"plc": task})
    pilot.initialize()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as callers:
            request_ids = list(
                callers.map(
                    lambda value: pilot.call_srv_async(
                        "plc",
                        {"output": value},
                        MoveSyncOption.no_sync(),
                        idempotency_key="cycle-42/output-1",
                    ),
                    range(20),
                )
            )
        assert request_ids == [request_ids[0]] * 20
        result = pilot.wait_request(request_ids[0], timeout=0.5)
        assert result.idempotency_key == "cycle-42/output-1"
        assert task.call_count == 1
    finally:
        pilot.stop()


def test_idempotency_key_is_service_scoped_and_released_when_pruned():
    first = CountingEchoTask()
    second = CountingEchoTask()
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]), {"camera": first, "plc": second}
    )
    pilot.initialize()
    try:
        camera_id = pilot.call_srv_async(
            "camera", sync_option=MoveSyncOption.no_sync(), idempotency_key="same"
        )
        plc_id = pilot.call_srv_async(
            "plc", sync_option=MoveSyncOption.no_sync(), idempotency_key="same"
        )
        pilot.wait_requests([camera_id, plc_id], timeout=0.5)
        assert camera_id != plc_id
        assert pilot.prune_completed_requests(keep_last=0) == 2
        replacement_id = pilot.call_srv_async(
            "camera", sync_option=MoveSyncOption.no_sync(), idempotency_key="same"
        )
        assert replacement_id not in {camera_id, plc_id}
        pilot.wait_request(replacement_id, timeout=0.5)
        assert first.call_count == 2
    finally:
        pilot.stop()


@pytest.mark.parametrize(
    "key, expected", [("", ValueError), (1, TypeError), (True, TypeError)]
)
def test_idempotency_key_validation(key, expected):
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"plc": EchoTask()})
    pilot.initialize()
    try:
        with pytest.raises(expected, match="idempotency_key"):
            pilot.call_srv_async("plc", idempotency_key=key)
    finally:
        pilot.stop()


def test_cancel_all_requests_can_filter_by_service():
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]),
        {"camera": EchoTask(), "plc": EchoTask()},
    )
    pilot.initialize()
    try:
        camera_id = pilot.call_srv_async(
            "camera", sync_option=MoveSyncOption.sync_w_explicit_id(999)
        )
        plc_id = pilot.call_srv_async(
            "plc", sync_option=MoveSyncOption.sync_w_explicit_id(999)
        )
        deadline = time.monotonic() + 0.5
        while pilot.query_request(plc_id).status is not RequestStatus.WAITING_FOR_MOVE:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert pilot.cancel_all_requests("camera") == [camera_id]
        assert pilot.query_request(camera_id).status is RequestStatus.CANCELLED
        assert not pilot.query_request(plc_id).status.terminal
        assert pilot.cancel_all_requests() == [plc_id]
        with pytest.raises(KeyError, match="Unknown service"):
            pilot.cancel_all_requests("missing")
    finally:
        pilot.stop()


def test_task_status_rejects_unknown_service():
    pilot = TaskPilot(SimulatedRobotTask([0.0]))
    with pytest.raises(KeyError, match="Unknown service"):
        pilot.query_task_status("missing")


def test_service_descriptions_are_json_compatible_and_do_not_poll_devices():
    status_calls = {"count": 0}

    class DiscoverableTask(EchoTask):
        def peek_status(self):
            status_calls["count"] += 1
            return {"health": "online"}

    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": DiscoverableTask()})
    descriptions = pilot.describe_services()
    json.dumps(descriptions)
    assert status_calls["count"] == 0
    assert descriptions["echo"]["service_name"] == "echo"
    assert descriptions["echo"]["task_type"] == "DiscoverableTask"
    assert descriptions["echo"]["capabilities"]["status"]


def test_all_task_statuses_isolates_device_monitoring_failures():
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]),
        {"echo": EchoTask(), "offline": StatusFailingTask()},
    )
    statuses = pilot.query_all_task_statuses()
    assert statuses["echo"]["execution"] == "in_place"
    assert statuses["offline"] == {
        "available": False,
        "health": "offline",
        "observed_at": pytest.approx(time.time(), abs=1.0),
        "error": "device offline",
    }


def test_health_snapshot_combines_scheduler_robot_requests_and_peripherals():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "echo", {"value": 3}, MoveSyncOption.no_sync()
        )
        pilot.wait_request(request_id, timeout=0.5)
        health = pilot.query_health()
        json.dumps(health)
        assert health["healthy"]
        assert health["generated_at"] > 0
        assert health["pilot"] == {
            "initialized": True,
            "scheduler_alive": True,
            "services": ["echo"],
        }
        assert health["robot"]["available"]
        assert health["robot"]["joint_positions"] == [0.0]
        assert health["requests"]["total"] == 1
        assert health["requests"]["succeeded"] == 1
        assert health["peripherals"]["echo"]["retained_results"] == 1
    finally:
        pilot.stop()


def test_health_snapshot_reports_offline_peripheral_without_raising():
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]), {"offline": StatusFailingTask()}
    )
    pilot.initialize()
    try:
        health = pilot.query_health()
        assert not health["healthy"]
        assert health["robot"]["available"]
        assert health["peripherals"]["offline"]["error"] == "device offline"
    finally:
        pilot.stop()


def test_health_snapshot_is_not_healthy_while_device_connects():
    class ConnectingTask(EchoTask):
        def peek_status(self):
            return {"health": "connecting", "available": False}

    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"sensor": ConnectingTask()})
    pilot.initialize()
    try:
        health = pilot.query_health()
        assert not health["healthy"]
        assert health["peripherals"]["sensor"]["health"] == "connecting"
    finally:
        pilot.stop()


def test_health_snapshot_marks_stale_device_as_degraded():
    class StaleTask(EchoTask):
        def peek_status(self):
            return {"health": "online", "observed_at": time.time() - 10.0}

    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"sensor": StaleTask()})
    pilot.initialize()
    try:
        health = pilot.query_health(stale_after=1.0)
        sensor = health["peripherals"]["sensor"]
        assert sensor["stale"]
        assert sensor["health"] == "degraded"
        assert not health["healthy"]
    finally:
        pilot.stop()


@pytest.mark.parametrize(
    "status, message",
    [
        ({"health": "broken"}, "health state"),
        ({"observed_at": float("nan")}, "observed_at"),
    ],
)
def test_task_status_rejects_malformed_health_metadata(status, message):
    class MalformedStatusTask(EchoTask):
        def peek_status(self):
            return status

    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"device": MalformedStatusTask()})
    with pytest.raises(ValueError, match=message):
        pilot.query_task_status("device")


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_health_snapshot_rejects_invalid_stale_timeout(value):
    pilot = TaskPilot(SimulatedRobotTask([0.0]))
    with pytest.raises(ValueError, match="stale_after"):
        pilot.query_health(stale_after=value)


def test_wait_health_change_returns_new_snapshot_without_polling():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    try:
        revision = pilot.health_revision
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            waiter = executor.submit(pilot.wait_health_change, revision, 0.5)
            request_id = pilot.call_srv_async(
                "echo", {"value": 1}, MoveSyncOption.no_sync()
            )
            snapshot = waiter.result(timeout=0.5)
        assert snapshot is not None
        assert snapshot["revision"] > revision
        pilot.wait_request(request_id, timeout=0.5)
        assert pilot.wait_health_change(pilot.health_revision, timeout=0) is None
    finally:
        pilot.stop()


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_wait_health_change_rejects_invalid_revision(value):
    pilot = TaskPilot(SimulatedRobotTask([0.0]))
    expected = TypeError if value in (1.5, True) else ValueError
    with pytest.raises(expected, match="after_revision"):
        pilot.wait_health_change(value, timeout=0)


def test_wait_health_change_rejects_future_revision():
    pilot = TaskPilot(SimulatedRobotTask([0.0]))
    with pytest.raises(ValueError, match="newer"):
        pilot.wait_health_change(pilot.health_revision + 1, timeout=0)


def test_initialization_failure_rolls_back_started_tasks():
    first = LifecycleTask()
    failing = LifecycleTask(fail_initialize=True)
    robot = SimulatedRobotTask([0.0])
    pilot = TaskPilot(robot, {"first": first, "failing": failing})
    with pytest.raises(RuntimeError, match="initialization failed"):
        pilot.initialize()
    assert first.stopped
    assert failing.stopped
    assert first._completion_callback is None
    assert failing._completion_callback is None
    assert robot._thread is not None and not robot._thread.is_alive()


def test_robot_initialization_failure_is_rolled_back():
    robot = PartiallyFailingRobot([0.0])
    pilot = TaskPilot(robot)
    with pytest.raises(RuntimeError, match="robot initialization failed"):
        pilot.initialize()
    assert robot._thread is not None and not robot._thread.is_alive()


def test_callback_registration_failure_rolls_back_all_tasks():
    robot = CallbackFailingRobot([0.0])
    task = LifecycleTask()
    pilot = TaskPilot(robot, {"task": task})
    with pytest.raises(RuntimeError, match="callback registration failed"):
        pilot.initialize()
    assert task.stopped
    assert task._completion_callback is None
    assert robot._thread is not None and not robot._thread.is_alive()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -0.1, True])
def test_invalid_poll_interval_is_rejected(value):
    with pytest.raises(ValueError, match="poll_interval"):
        TaskPilot(SimulatedRobotTask([0.0]), poll_interval=value)


def test_pilot_validates_capacity_service_names_and_copies_task_map():
    with pytest.raises(TypeError, match="timeline_capacity"):
        TaskPilot(SimulatedRobotTask([0.0]), timeline_capacity=1.5)
    with pytest.raises(TypeError, match="timeline_capacity"):
        TaskPilot(SimulatedRobotTask([0.0]), timeline_capacity=True)
    with pytest.raises(ValueError, match="service names"):
        TaskPilot(SimulatedRobotTask([0.0]), {"": EchoTask()})

    tasks = {"echo": EchoTask()}
    pilot = TaskPilot(SimulatedRobotTask([0.0]), tasks)
    tasks.clear()
    assert "echo" in pilot.task_map
    exposed_tasks = pilot.task_map
    exposed_tasks.clear()
    assert pilot.service_names == ("echo",)


def test_pilot_context_manager_owns_lifecycle():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    assert not pilot.is_running
    with pilot as running:
        assert running is pilot
        assert pilot.is_running
    assert not pilot.is_running


def test_concurrent_lifecycle_calls_are_serialized():
    robot = CountingRobot()
    pilot = TaskPilot(robot, {"echo": EchoTask()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        starts = [executor.submit(pilot.initialize) for _ in range(2)]
        for future in starts:
            future.result(timeout=0.5)
    assert pilot.is_running
    assert robot.initialize_count == 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        stops = [executor.submit(pilot.stop) for _ in range(2)]
        for future in stops:
            future.result(timeout=0.5)
    assert not pilot.is_running
    assert robot.stop_count == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, True])
def test_invalid_request_timeout_is_rejected(value):
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async("echo", sync_option=MoveSyncOption.no_sync())
        with pytest.raises(ValueError, match="timeout"):
            pilot.wait_request(request_id, timeout=value)
    finally:
        pilot.stop()


@pytest.mark.parametrize("request_id", [True, 1.5, "1"])
def test_request_apis_reject_non_integer_ids(request_id):
    pilot = TaskPilot(SimulatedRobotTask([0.0]))
    with pytest.raises(TypeError, match="request_id"):
        pilot.query_request(request_id)
    with pytest.raises(TypeError, match="request_id"):
        pilot.wait_request(request_id)
    with pytest.raises(TypeError, match="request_id"):
        pilot.cancel_request(request_id)
    with pytest.raises(TypeError, match="request_id"):
        pilot.timeline(request_id)


def test_request_apis_reject_negative_ids():
    pilot = TaskPilot(SimulatedRobotTask([0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        pilot.query_request(-1)
    with pytest.raises(ValueError, match="non-negative"):
        pilot.wait_request(-1)
    with pytest.raises(ValueError, match="non-negative"):
        pilot.cancel_request(-1)
    with pytest.raises(ValueError, match="non-negative"):
        pilot.timeline(-1)


def test_request_results_are_defensive_copies():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "echo",
            {"nested": {"value": 1}},
            sync_option=MoveSyncOption.no_sync(),
        )
        result = pilot.wait_request(request_id, timeout=0.5)
        result.content["nested"]["value"] = 99
        assert pilot.query_request(request_id).content["nested"]["value"] == 1

        queried = pilot.query_request(request_id)
        queried.content["nested"]["value"] = 42
        assert pilot.query_request(request_id).content["nested"]["value"] == 1
    finally:
        pilot.stop()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -0.1])
def test_simulated_robot_rejects_invalid_motion_interval(value):
    robot = SimulatedRobotTask([0.0])
    assert robot.move_joint_trajectory_async([[0.1]], interval=value) == -1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.1", True])
def test_simulated_robot_rejects_non_finite_or_non_numeric_joints(value):
    assert SimulatedRobotTask([0.0]).move_joint_trajectory_async([[value]]) == -1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.0", True])
def test_simulated_robot_rejects_invalid_initial_joints(value):
    with pytest.raises(ValueError, match="finite real"):
        SimulatedRobotTask([value])


@pytest.mark.parametrize(
    "time_out,interval",
    [
        (float("nan"), 0.01),
        (float("inf"), 0.01),
        (0.1, 0.0),
        (0.1, -0.01),
        (0.1, float("nan")),
    ],
)
def test_simulated_wait_rejects_invalid_timing(time_out, interval):
    with pytest.raises(ValueError):
        SimulatedRobotTask([0.0]).wait_move(time_out=time_out, interval=interval)


@pytest.mark.parametrize(
    "timeout", [-0.1, float("nan"), float("inf"), "0.1", True]
)
def test_simulated_stop_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        SimulatedRobotTask([0.0]).stop(timeout=timeout)


def test_simulated_wait_timeout_is_not_extended_by_poll_interval():
    robot = SimulatedRobotTask([0.0])
    robot.move_joint_trajectory_async([[0.1]], interval=0.01)
    started = time.monotonic()
    assert not robot.wait_move(time_out=0.02, interval=1.0)
    assert time.monotonic() - started < 0.1


def test_simulated_wait_is_woken_by_completion_before_poll_interval():
    robot = SimulatedRobotTask([0.0])
    robot.initialize()
    try:
        robot.move_joint_trajectory_async([[0.1]], interval=0.002)
        started = time.monotonic()
        assert robot.wait_move(time_out=0.5, interval=1.0)
        assert time.monotonic() - started < 0.1
    finally:
        robot.stop()


def test_simulated_robot_concurrent_lifecycle_calls_are_serialized():
    robot = SimulatedRobotTask([0.0])
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(robot.initialize) for _ in range(16)]
        for future in futures:
            future.result(timeout=0.5)
        thread = robot._thread
        assert thread is not None and thread.is_alive()

        futures = [executor.submit(robot.stop) for _ in range(16)]
        for future in futures:
            future.result(timeout=0.5)
    assert robot._thread is thread
    assert not thread.is_alive()


@pytest.mark.parametrize("timeout", [-0.1, float("nan"), float("inf"), True])
def test_stop_rejects_invalid_timeout(timeout):
    pilot = TaskPilot(SimulatedRobotTask([0.0]))
    with pytest.raises(ValueError, match="timeout"):
        pilot.stop(timeout=timeout)


def test_request_failure_is_observable():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"fail": FailingTask()})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async("fail", sync_option=MoveSyncOption.no_sync())
        result = pilot.wait_request(request_id, timeout=0.5)
        assert result.status is RequestStatus.FAILED
        assert "broken peripheral" in result.error
    finally:
        pilot.stop()


def test_wait_timeout_and_cancel_waiting_request():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "echo", sync_option=MoveSyncOption.sync_w_explicit_id(99)
        )
        with pytest.raises(TimeoutError):
            pilot.wait_request(request_id, timeout=0.02)
        assert pilot.cancel_request(request_id)
        assert pilot.wait_request(request_id).status is RequestStatus.CANCELLED
        assert not pilot.cancel_request(request_id)
    finally:
        pilot.stop()


def test_queued_request_snapshots_nested_content():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    try:
        content = {"nested": {"value": 1}}
        _, move_end = pilot.move_joint_trajectory_async([[0.1]], interval=0.03)
        request_id = pilot.call_srv_async(
            "echo",
            content,
            MoveSyncOption.sync_w_explicit_id(move_end - 1),
        )
        content["nested"]["value"] = 2
        result = pilot.wait_request(request_id, timeout=0.5)
        assert result.content["nested"]["value"] == 1
    finally:
        pilot.stop()


def test_stop_cancels_requests_and_unblocks_waiters():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    request_id = pilot.call_srv_async(
        "echo", sync_option=MoveSyncOption.sync_w_explicit_id(99)
    )
    outcome = []
    waiter = threading.Thread(target=lambda: outcome.append(pilot.wait_request(request_id)))
    waiter.start()
    pilot.stop()
    waiter.join(timeout=0.5)
    assert not waiter.is_alive()
    assert outcome[0].status is RequestStatus.CANCELLED


def test_stop_does_not_allow_request_to_enqueue_after_cancellation_pass():
    preparing = threading.Event()
    release = threading.Event()

    class BlockingCopyDict(dict):
        def __deepcopy__(self, memo):
            preparing.set()
            assert release.wait(0.5)
            return dict(self)

    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        caller = executor.submit(
            pilot.call_srv_async,
            "echo",
            BlockingCopyDict(value=1),
            MoveSyncOption.no_sync(),
        )
        assert preparing.wait(0.5)
        pilot.stop()
        release.set()
        with pytest.raises(RuntimeError, match="stopped"):
            caller.result(timeout=0.5)
    assert pilot.timeline() == []


def test_stop_does_not_wait_forever_for_running_callback():
    release = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    task = BlockingTask(release)
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]),
        {"blocking": task},
        executor_owned=executor,
    )
    pilot.initialize()
    request_id = pilot.call_srv_async(
        "blocking", sync_option=MoveSyncOption.no_sync()
    )
    deadline = time.monotonic() + 0.5
    while pilot.query_request(request_id).status is not RequestStatus.RUNNING:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert task.started.wait(0.5)
    started = time.monotonic()
    try:
        pilot.stop(timeout=0.02)
        assert time.monotonic() - started < 0.1
        assert pilot.query_request(request_id).status is RequestStatus.CANCELLED
        with pytest.raises(RuntimeError):
            executor.submit(lambda: None)
    finally:
        release.set()


def test_cancel_running_thread_pool_request():
    blocker = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(blocker.wait)
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]),
        {"slow": SlowTask()},
        executor_owned=executor,
    )
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async("slow", sync_option=MoveSyncOption.no_sync())
        deadline = time.monotonic() + 0.5
        while pilot.query_request(request_id).status is not RequestStatus.RUNNING:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert pilot.cancel_request(request_id)
        assert pilot.wait_request(request_id).status is RequestStatus.CANCELLED
    finally:
        blocker.set()
        pilot.stop()


def test_repeated_queued_cancellation_stays_cancelled():
    for _ in range(25):
        blocker = threading.Event()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(blocker.wait)
        pilot = TaskPilot(
            SimulatedRobotTask([0.0]),
            {"slow": SlowTask()},
            executor_owned=executor,
            poll_interval=0.001,
        )
        pilot.initialize()
        try:
            request_id = pilot.call_srv_async(
                "slow", sync_option=MoveSyncOption.no_sync()
            )
            deadline = time.monotonic() + 0.5
            while pilot.query_request(request_id).status is not RequestStatus.RUNNING:
                assert time.monotonic() < deadline
                time.sleep(0.001)
            assert pilot.cancel_request(request_id)
            assert pilot.wait_request(request_id).status is RequestStatus.CANCELLED
        finally:
            blocker.set()
            pilot.stop()


def test_thread_pool_completion_wakes_scheduler():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]),
        {"delayed": DelayedTask()},
        executor_owned=executor,
        poll_interval=1.0,
    )
    pilot.initialize()
    try:
        started = time.monotonic()
        request_id = pilot.call_srv_async(
            "delayed", {"done": True}, MoveSyncOption.no_sync()
        )
        result = pilot.wait_request(request_id, timeout=0.5)
        assert result.status is RequestStatus.SUCCEEDED
        assert time.monotonic() - started < 0.5
    finally:
        pilot.stop()


def test_thread_pool_failure_preserves_error():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]),
        {"fail": AsyncFailingTask()},
        executor_owned=executor,
    )
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async("fail", sync_option=MoveSyncOption.no_sync())
        result = pilot.wait_request(request_id, timeout=0.5)
        assert result.status is RequestStatus.FAILED
        assert result.error == "async failure"
    finally:
        pilot.stop()


def test_simulated_pick_and_place_flow_is_event_driven():
    pilot = TaskPilot(
        SimulatedRobotTask([0.0, 0.0]),
        {"gripper": EchoTask()},
        poll_interval=1.0,
    )
    pilot.initialize()
    try:
        _, move_end = pilot.move_joint_trajectory_async(
            [[0.2, 0.1], [0.8, 0.4]], interval=0.01
        )
        started = time.monotonic()
        request_id = pilot.call_srv_async(
            "gripper",
            {"action": "close"},
            MoveSyncOption.sync_w_explicit_id(move_end - 1),
        )
        result = pilot.wait_request(request_id, timeout=0.5)
        assert result.status is RequestStatus.SUCCEEDED
        assert time.monotonic() - started < 0.5
        assert pilot.query_robot_state().jps == [0.8, 0.4]
        statuses = [event.status for event in pilot.timeline(request_id)]
        assert RequestStatus.WAITING_FOR_MOVE in statuses
    finally:
        pilot.stop()


def test_simulated_robot_tracks_segment_move_ids():
    robot = SimulatedRobotTask([0.0])
    robot.initialize()
    try:
        move_begin = robot.move_joint_trajectory_async(
            [[0.1], [0.2], [0.3]],
            interval=0.005,
            endpoint_index=[1, 3],
        )
        assert move_begin == 0
        assert robot.query_state().latest_sent_id == 1
        assert robot.wait_move(time_out=0.5)
        state = robot.query_state()
        assert state.latest_finished_id == 1
        assert state.jps == [0.3]
    finally:
        robot.stop()


def test_simulated_robot_survives_failing_state_callback():
    robot = SimulatedRobotTask([0.0])

    def fail_callback():
        raise RuntimeError("callback failure")

    robot.set_state_change_callback(fail_callback)
    robot.initialize()
    try:
        robot.move_joint_trajectory_async([[0.1], [0.2]], interval=0.002)
        assert robot.wait_move(time_out=0.2)
        assert robot.query_state().jps == [0.2]
        assert robot._thread.is_alive()
    finally:
        robot.stop()


def test_simulated_restart_does_not_execute_motion_queued_before_stop():
    robot = SimulatedRobotTask([0.0])
    robot.initialize()
    robot.move_joint_trajectory_async([[0.1], [0.2]], interval=0.1)
    robot.move_joint_trajectory_async([[0.9]], interval=0.002)
    time.sleep(0.01)
    robot.stop()
    stopped_position = robot.query_state().jps.copy()
    robot.initialize()
    try:
        time.sleep(0.05)
        assert robot.query_state().jps == stopped_position
    finally:
        robot.stop()


@pytest.mark.parametrize("endpoints", [[], [1.0], [True]])
def test_simulated_robot_rejects_invalid_endpoint_types(endpoints):
    robot = SimulatedRobotTask([0.0])
    assert robot.move_joint_trajectory_async([[0.1]], endpoint_index=endpoints) == -1


def test_timeline_capacity_is_bounded():
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]), {"echo": EchoTask()}, timeline_capacity=3
    )
    pilot.initialize()
    try:
        first = pilot.call_srv_async("echo", sync_option=MoveSyncOption.no_sync())
        pilot.wait_request(first, timeout=0.5)
        second = pilot.call_srv_async("echo", sync_option=MoveSyncOption.no_sync())
        pilot.wait_request(second, timeout=0.5)
        assert len(pilot.timeline()) == 3
        assert pilot.timeline()[-1].request_id == second
    finally:
        pilot.stop()


def test_evicted_task_results_do_not_leave_running_requests():
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]),
        {"echo": EchoTask(max_result_count=1)},
    )
    pilot.initialize()
    try:
        request_ids = [
            pilot.call_srv_async("echo", sync_option=MoveSyncOption.no_sync())
            for _ in range(3)
        ]
        results = [
            pilot.wait_request(request_id, timeout=0.5)
            for request_id in request_ids
        ]
        assert [result.status for result in results] == [
            RequestStatus.FAILED,
            RequestStatus.FAILED,
            RequestStatus.SUCCEEDED,
        ]
        assert "flushed" in results[0].error
    finally:
        pilot.stop()


def test_completed_request_history_can_be_pruned_with_task_results():
    task = EchoTask()
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": task})
    pilot.initialize()
    try:
        request_ids = []
        for value in range(5):
            request_id = pilot.call_srv_async(
                "echo", {"value": value}, MoveSyncOption.no_sync()
            )
            pilot.wait_request(request_id, timeout=0.5)
            request_ids.append(request_id)

        assert pilot.prune_completed_requests(keep_last=2) == 3
        for request_id in request_ids[:3]:
            with pytest.raises(KeyError):
                pilot.query_request(request_id)
            assert request_id not in task._result_map
        assert pilot.query_request(request_ids[-1]).status is RequestStatus.SUCCEEDED
        assert pilot.prune_completed_requests(keep_last=2) == 0
    finally:
        pilot.stop()


def test_pruning_cancelled_waiting_request_does_not_break_scheduler():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    try:
        cancelled_id = pilot.call_srv_async(
            "echo", sync_option=MoveSyncOption.sync_w_explicit_id(99)
        )
        deadline = time.monotonic() + 0.5
        while (
            pilot.query_request(cancelled_id).status
            is not RequestStatus.WAITING_FOR_MOVE
        ):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert pilot.cancel_request(cancelled_id)
        assert pilot.prune_completed_requests(keep_last=0) == 1

        next_id = pilot.call_srv_async("echo", sync_option=MoveSyncOption.no_sync())
        assert pilot.wait_request(next_id, timeout=0.5).status is RequestStatus.SUCCEEDED
    finally:
        pilot.stop()


@pytest.mark.parametrize("keep_last", [-1, 1.5, True])
def test_request_history_pruning_validates_keep_last(keep_last):
    pilot = TaskPilot(SimulatedRobotTask([0.0]))
    expected = ValueError if keep_last == -1 else TypeError
    with pytest.raises(expected):
        pilot.prune_completed_requests(keep_last)


def test_concurrent_callers_get_unique_ordered_requests():
    pilot = TaskPilot(SimulatedRobotTask([0.0]), {"echo": EchoTask()})
    pilot.initialize()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as callers:
            request_ids = list(
                callers.map(
                    lambda value: pilot.call_srv_async(
                        "echo", {"value": value}, MoveSyncOption.no_sync()
                    ),
                    range(50),
                )
            )
        assert sorted(request_ids) == list(range(50))
        results = [pilot.wait_request(request_id, timeout=1.0) for request_id in request_ids]
        assert all(result.status is RequestStatus.SUCCEEDED for result in results)
    finally:
        pilot.stop()
