import concurrent.futures
import importlib
import threading
import time

import pytest

pytest.importorskip("viser")
pytest.importorskip("yourdfpy")

from examples.viser.common import PICK_Q, PLACE_Q, ViserDemoRuntime
from integrations.viser.viser_modular_robot_task import (
    ViserGripperTask,
    ViserModularReducedRobotTask,
)
from orchestrion import MoveSyncOption, RequestStatus, TaskPilot
from orchestrion.tasks import ModularReducedRobotTask, SubModuleTask
from orchestrion.utils.types import PeekResponseResultType

DEMO_MODULES = [
    "examples.viser.01_joint_sweep",
    "examples.viser.02_pick_and_place",
    "examples.viser.03_segmented_sync",
    "examples.viser.04_parallel_motion",
    "examples.viser.05_request_timeline",
    "examples.viser.06_interactive_controls",
    "examples.viser.07_motion_trail",
    "examples.viser.08_gripper_lab",
    "examples.viser.viser_modular_robot_task",
]


@pytest.mark.parametrize("module_name", DEMO_MODULES)
def test_viser_demo_imports(module_name):
    module = importlib.import_module(module_name)
    assert callable(module.workflow)


def test_viser_robot_accepts_configurable_scheduler_interval():
    robot = ViserModularReducedRobotTask(
        [0.0, 0.0],
        {},
        SubModuleTask("arm", 0, 1),
        [SubModuleTask("gripper", 1, 1)],
        interval=0.003,
    )
    assert robot.scheduler_interval == 0.003


def test_runtime_waits_until_first_client_connects():
    callbacks = []

    class FakeServer:
        def get_clients(self):
            return {}

        def on_client_connect(self, callback):
            callbacks.append(callback)
            return callback

    runtime = object.__new__(ViserDemoRuntime)
    runtime.server = FakeServer()
    runtime._monitor_stop = threading.Event()
    runtime._demo_status = type("Status", (), {"content": ""})()
    waiter = threading.Thread(target=runtime.wait_for_client)
    waiter.start()
    deadline = time.monotonic() + 0.5
    while not callbacks:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert waiter.is_alive()
    callbacks[0](object())
    waiter.join(timeout=0.5)
    assert not waiter.is_alive()
    assert "client connected" in runtime._demo_status.content


class VisibleStub:
    def __init__(self, visible):
        self.visible = visible


class PayloadRobotStub:
    def __init__(self, joints, gripper_position):
        self.joints = list(joints) + [gripper_position]
        self.gripper_position = gripper_position

    def query_state(self):
        return type("State", (), {"jps": self.joints})()

    def query_submodule_state(self, name):
        return type("State", (), {"positions": [self.gripper_position]})()


def make_payload_runtime(joints, gripper_position, attached=False):
    runtime = object.__new__(ViserDemoRuntime)
    runtime.robot = PayloadRobotStub(joints, gripper_position)
    runtime._pick_object = VisibleStub(True)
    runtime._carried_object = VisibleStub(attached)
    runtime._placed_object = VisibleStub(False)
    runtime._payload_attached = attached
    runtime._payload_status = type("Status", (), {"content": ""})()
    return runtime


def test_payload_attaches_only_at_pick_with_closed_gripper():
    runtime = make_payload_runtime(PICK_Q, 0.72)
    runtime.attach_payload()
    assert runtime._payload_attached
    assert not runtime._pick_object.visible
    assert runtime._carried_object.visible

    runtime = make_payload_runtime(PICK_Q, 0.2)
    with pytest.raises(RuntimeError, match="closed"):
        runtime.attach_payload()


def test_payload_releases_only_at_place_with_open_gripper():
    runtime = make_payload_runtime(PLACE_Q, 0.0, attached=True)
    runtime.release_payload()
    assert not runtime._payload_attached
    assert not runtime._carried_object.visible
    assert runtime._placed_object.visible

    runtime = make_payload_runtime(PICK_Q, 0.0, attached=True)
    with pytest.raises(RuntimeError, match="place pose"):
        runtime.release_payload()

    runtime = make_payload_runtime(PLACE_Q, 0.4, attached=True)
    with pytest.raises(RuntimeError, match="open"):
        runtime.release_payload()


class RobotStub:
    def __init__(self):
        self.requests = []

    scheduler_interval = 0.01

    def query_submodule_state(self, name):
        return type("State", (), {"positions": [0.2]})()

    def move_submodule_trajectory_async(self, name, trajectory, interval):
        self.requests.append((name, trajectory, interval))
        return 4

    def wait_submodule_move(self, name, move_id, timeout):
        return True

    def cancel_submodule_move(self, name, move_id):
        return True


def test_viser_gripper_task_reports_result():
    robot = RobotStub()
    task = ViserGripperTask(robot)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        assert task.invoke_async(0, {"action": "close"})
        deadline = time.monotonic() + 0.5
        while task.peek_response(0).result_type is not PeekResponseResultType.ResponseFound:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        response = task.peek_response(0)
    assert response.result_type is PeekResponseResultType.ResponseFound
    assert response.content == {
        "request_id": 0,
        "action": "close",
        "position": 0.72,
        "move_id": 4,
    }
    assert robot.requests[0][0] == "gripper"
    assert robot.requests[0][1][0] == [pytest.approx(0.2)]
    assert robot.requests[0][1][-1] == [pytest.approx(0.72)]


def test_duplicate_request_id_does_not_corrupt_queued_sequence():
    robot = RobotStub()
    task = ViserGripperTask(robot)
    release_worker = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(release_worker.wait)
        task.initialize(executor=executor)
        assert task.invoke_async(0, {"action": "close"})
        assert not task.invoke_async(0, {"action": "open"})
        release_worker.set()
        blocker.result(timeout=0.5)
        deadline = time.monotonic() + 0.5
        while task.peek_response(0).result_type is not PeekResponseResultType.ResponseFound:
            assert time.monotonic() < deadline
            time.sleep(0.005)
    assert robot.requests[0][1][-1] == [pytest.approx(0.72)]


def test_cancel_rejects_unknown_and_completed_requests():
    robot = RobotStub()
    task = ViserGripperTask(robot)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        assert not task.cancel_request(99)
        assert task.invoke_async(0, {"action": "close"})
        deadline = time.monotonic() + 0.5
        while task.peek_response(0).result_type is not PeekResponseResultType.ResponseFound:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert not task.cancel_request(0)


def test_gripper_task_can_restart_after_stopping_queued_request():
    robot = RobotStub()
    task = ViserGripperTask(robot)
    release_worker = threading.Event()
    first_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    blocker = first_executor.submit(release_worker.wait)
    task.initialize(executor=first_executor)
    assert task.invoke_async(0, {"action": "close"})
    task.stop()
    release_worker.set()
    blocker.result(timeout=0.5)
    first_executor.shutdown()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as second_executor:
        task.initialize(executor=second_executor)
        assert task.invoke_async(1, {"action": "open"})
        deadline = time.monotonic() + 0.5
        while task.peek_response(1).result_type is not PeekResponseResultType.ResponseFound:
            assert time.monotonic() < deadline
            time.sleep(0.005)


class ModularRobotStub(ModularReducedRobotTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodule_trajectories = []

    def _on_robot_state_bg_thread(self, jps, jps_move_id):
        pass

    def _on_initial_state_bg_thread(self, jps):
        pass

    def move_submodule_trajectory_async(
        self, submodule_name, motion_target, interval=0.01
    ):
        self.submodule_trajectories.append(motion_target)
        return super().move_submodule_trajectory_async(
            submodule_name, motion_target, interval
        )


def make_gripper_robot():
    return ModularRobotStub(
        [0.0, 0.0],
        SubModuleTask("arm", 0, 1),
        [SubModuleTask("gripper", 1, 1)],
        interval=0.002,
    )


def test_gripper_request_finishes_at_target_position():
    robot = make_gripper_robot()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    pilot = TaskPilot(
        robot, {"gripper": ViserGripperTask(robot)}, executor_owned=executor
    )
    pilot.initialize()
    try:
        close_id = pilot.call_srv_async(
            "gripper",
            {"action": "close", "speed": 100.0},
            MoveSyncOption.no_sync(),
        )
        result = pilot.wait_request(close_id, timeout=0.5)
        assert result.status is RequestStatus.SUCCEEDED
        assert robot.query_submodule_state("gripper").positions == [pytest.approx(0.72)]

        open_id = pilot.call_srv_async(
            "gripper",
            {"position": 0.25, "speed": 100.0},
            MoveSyncOption.no_sync(),
        )
        pilot.wait_request(open_id, timeout=0.5)
        assert robot.query_submodule_state("gripper").positions == [pytest.approx(0.25)]
    finally:
        pilot.stop()


def test_modular_robot_completion_wakes_synchronized_gripper():
    robot = make_gripper_robot()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(
        robot,
        {"gripper": ViserGripperTask(robot)},
        executor_owned=executor,
        poll_interval=1.0,
    )
    pilot.initialize()
    try:
        _, move_end = pilot.move_joint_trajectory_async([[0.2]], interval=0.002)
        started = time.monotonic()
        request_id = pilot.call_srv_async(
            "gripper",
            {"action": "close", "speed": 100.0},
            MoveSyncOption.sync_w_explicit_id(move_end - 1),
        )
        assert pilot.wait_request(request_id, timeout=0.5).status is RequestStatus.SUCCEEDED
        assert time.monotonic() - started < 0.5
    finally:
        pilot.stop()


def test_gripper_request_can_be_cancelled_mid_motion():
    robot = make_gripper_robot()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(
        robot, {"gripper": ViserGripperTask(robot)}, executor_owned=executor
    )
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "gripper",
            {"action": "close", "speed": 0.2},
            MoveSyncOption.no_sync(),
        )
        deadline = time.monotonic() + 0.5
        while pilot.query_request(request_id).status is not RequestStatus.RUNNING:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        assert pilot.cancel_request(request_id)
        assert pilot.wait_request(request_id).status is RequestStatus.CANCELLED
    finally:
        pilot.stop()


def test_stopping_pilot_cancels_slow_gripper_promptly():
    robot = make_gripper_robot()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(
        robot, {"gripper": ViserGripperTask(robot)}, executor_owned=executor
    )
    pilot.initialize()
    request_id = pilot.call_srv_async(
        "gripper",
        {"action": "close", "speed": 0.05, "timeout": 30.0},
        MoveSyncOption.no_sync(),
    )
    deadline = time.monotonic() + 0.5
    while pilot.query_request(request_id).status is not RequestStatus.RUNNING:
        assert time.monotonic() < deadline
        time.sleep(0.002)
    started = time.monotonic()
    pilot.stop()
    assert time.monotonic() - started < 0.5
    assert pilot.query_request(request_id).status is RequestStatus.CANCELLED


@pytest.mark.parametrize(
    "content,error_text",
    [
        ({"position": -0.1}, "position"),
        ({"position": 0.8}, "position"),
        ({"action": "close", "speed": 0.0}, "speed"),
        ({"action": "close", "timeout": 0.0}, "timeout"),
        ({"action": "invalid"}, "action"),
    ],
)
def test_invalid_gripper_commands_fail_with_clear_error(content, error_text):
    robot = make_gripper_robot()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(
        robot, {"gripper": ViserGripperTask(robot)}, executor_owned=executor
    )
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "gripper", content, MoveSyncOption.no_sync()
        )
        result = pilot.wait_request(request_id, timeout=0.5)
        assert result.status is RequestStatus.FAILED
        assert error_text in result.error.lower()
    finally:
        pilot.stop()


def test_invalid_timeout_does_not_enqueue_motion():
    robot = make_gripper_robot()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(
        robot, {"gripper": ViserGripperTask(robot)}, executor_owned=executor
    )
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "gripper",
            {"action": "close", "timeout": float("nan")},
            MoveSyncOption.no_sync(),
        )
        assert pilot.wait_request(request_id, timeout=0.5).status is RequestStatus.FAILED
        assert robot.submodule_trajectories == []
    finally:
        pilot.stop()


def test_cancel_then_reverse_starts_from_actual_position():
    robot = make_gripper_robot()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(
        robot, {"gripper": ViserGripperTask(robot)}, executor_owned=executor
    )
    pilot.initialize()
    try:
        close_id = pilot.call_srv_async(
            "gripper",
            {"action": "close", "speed": 0.3},
            MoveSyncOption.no_sync(),
        )
        deadline = time.monotonic() + 0.5
        while robot.query_submodule_state("gripper").positions[0] <= 0.01:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        assert pilot.cancel_request(close_id)
        cancelled_at = robot.query_submodule_state("gripper").positions[0]

        open_id = pilot.call_srv_async(
            "gripper",
            {"action": "open", "speed": 100.0},
            MoveSyncOption.no_sync(),
        )
        assert pilot.wait_request(open_id, timeout=0.5).status is RequestStatus.SUCCEEDED
        reverse_trajectory = robot.submodule_trajectories[-1]
        assert reverse_trajectory[0][0] == pytest.approx(cancelled_at, abs=0.01)
        assert reverse_trajectory[-1] == [0.0]
    finally:
        pilot.stop()


def test_gripper_timeout_stops_underlying_motion():
    robot = make_gripper_robot()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(
        robot, {"gripper": ViserGripperTask(robot)}, executor_owned=executor
    )
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "gripper",
            {"action": "close", "speed": 0.05, "timeout": 0.02},
            MoveSyncOption.no_sync(),
        )
        result = pilot.wait_request(request_id, timeout=0.5)
        assert result.status is RequestStatus.FAILED
        position_after_failure = robot.query_submodule_state("gripper").positions[0]
        time.sleep(0.03)
        assert robot.query_submodule_state("gripper").positions[0] == pytest.approx(
            position_after_failure, abs=0.002
        )
        assert position_after_failure < 0.72
    finally:
        pilot.stop()


def test_concurrent_executor_preserves_gripper_command_order():
    robot = make_gripper_robot()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    pilot = TaskPilot(
        robot, {"gripper": ViserGripperTask(robot)}, executor_owned=executor
    )
    pilot.initialize()
    targets = [0.1, 0.6, 0.2, 0.7, 0.0] * 4
    try:
        request_ids = [
            pilot.call_srv_async(
                "gripper",
                {"position": target, "speed": 100.0},
                MoveSyncOption.no_sync(),
            )
            for target in targets
        ]
        results = [pilot.wait_request(request_id, timeout=2.0) for request_id in request_ids]
        assert all(result.status is RequestStatus.SUCCEEDED for result in results)
        assert [trajectory[-1][0] for trajectory in robot.submodule_trajectories] == [
            pytest.approx(target) for target in targets
        ]
        assert robot.query_submodule_state("gripper").positions == [0.0]
    finally:
        pilot.stop()


def test_cancelling_queued_gripper_command_does_not_block_following_command():
    robot = make_gripper_robot()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pilot = TaskPilot(
        robot, {"gripper": ViserGripperTask(robot)}, executor_owned=executor
    )
    pilot.initialize()
    try:
        first = pilot.call_srv_async(
            "gripper", {"position": 0.2, "speed": 0.5}, MoveSyncOption.no_sync()
        )
        cancelled = pilot.call_srv_async(
            "gripper", {"position": 0.7, "speed": 0.5}, MoveSyncOption.no_sync()
        )
        last = pilot.call_srv_async(
            "gripper", {"position": 0.1, "speed": 100.0}, MoveSyncOption.no_sync()
        )
        deadline = time.monotonic() + 0.5
        while pilot.query_request(cancelled).status is not RequestStatus.RUNNING:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        assert pilot.cancel_request(cancelled)
        assert pilot.wait_request(first, timeout=1.0).status is RequestStatus.SUCCEEDED
        assert pilot.wait_request(last, timeout=1.0).status is RequestStatus.SUCCEEDED
        assert pilot.query_request(cancelled).status is RequestStatus.CANCELLED
        assert robot.query_submodule_state("gripper").positions == [pytest.approx(0.1)]
    finally:
        pilot.stop()
