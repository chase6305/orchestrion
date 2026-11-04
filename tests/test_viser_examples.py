import concurrent.futures
import importlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yourdfpy")
pytest.importorskip("viser")

import numpy as np
import yourdfpy
from viser.transforms import SO3

from examples.dual_arm.viser import (
    DualArmViserRuntime,
    ReplayFrame,
    replay_frame_index,
    visual_arm_configuration,
    world_ee_position,
)
from examples.viser.__main__ import main as viser_demo_main
from examples.viser.common import (
    HOME_Q,
    PAYLOAD_DIMENSIONS,
    PAYLOAD_GRASP_POSITION,
    PAYLOAD_OFFSET,
    PICK_ABOVE_Q,
    PICK_Q,
    PLACE_ABOVE_Q,
    PLACE_Q,
    ViserDemoRuntime,
    interpolate,
)
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


def test_viser_launcher_lists_demos(capsys):
    viser_demo_main(["--list"])
    output = capsys.readouterr().out
    assert "02  Pick and Place" in output
    assert "08  Gripper Laboratory" in output


def test_viser_launcher_forwards_help_to_selected_demo(capsys):
    with pytest.raises(SystemExit) as error:
        viser_demo_main(["02", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "02 · Pick and Place" in output
    assert "--no-wait-for-client" in output


def test_dual_arm_visual_configuration_adds_stable_wrist_pose():
    configuration = visual_arm_configuration([0.2, -0.3, 0.4])
    assert configuration.shape == (6,)
    assert np.all(np.isfinite(configuration))


def test_dual_arm_visual_handoff_poses_are_mirrored():
    left = visual_arm_configuration([0.35, -0.05, -0.25], "left")
    right = visual_arm_configuration([0.35, 0.05, 0.25], "right")
    assert left.shape == right.shape == (6,)


def test_dual_arm_visual_handoff_end_effectors_meet():
    urdf_path = Path(__file__).resolve().parents[1] / "assets" / "UR5" / "UR5.urdf"
    left_urdf = yourdfpy.URDF.load(str(urdf_path))
    right_urdf = yourdfpy.URDF.load(str(urdf_path))
    left = visual_arm_configuration([0.35, -0.05, -0.25], "left")
    right = visual_arm_configuration([0.35, 0.05, 0.25], "right")
    payload_offset = np.array([0.0, 0.0, 0.08])
    left_ee = world_ee_position(left_urdf, left, "left", payload_offset)
    right_ee = world_ee_position(right_urdf, right, "right", payload_offset)
    assert np.linalg.norm(left_ee - right_ee) < 0.002
    left_rotation = left_urdf.get_transform("ee_link", "base_link")[:3, :3]
    right_rotation = right_urdf.get_transform("ee_link", "base_link")[:3, :3]
    assert np.dot(left_rotation[:, 2], right_rotation[:, 2]) > 0.999
    assert np.dot(left_rotation[:, 0], right_rotation[:, 0]) < -0.999


@pytest.mark.parametrize(
    "positions", ([0.0, 0.0], [0.0, 0.0, float("nan")])
)
def test_dual_arm_visual_configuration_rejects_invalid_state(positions):
    with pytest.raises(ValueError, match="three finite"):
        visual_arm_configuration(positions)


def test_dual_arm_visual_configuration_rejects_unknown_side():
    with pytest.raises(ValueError, match="side"):
        visual_arm_configuration([0.0, 0.0, 0.0], "center")


def test_dual_arm_replay_selects_frame_at_or_before_cursor():
    frames = [
        ReplayFrame(value, (), (), 0.0, 0.0, None, "table", "move")
        for value in (0.0, 0.2, 0.7)
    ]
    assert replay_frame_index(frames, -1.0) == 0
    assert replay_frame_index(frames, 0.19) == 0
    assert replay_frame_index(frames, 0.2) == 1
    assert replay_frame_index(frames, 99.0) == 2


def test_dual_arm_replay_rejects_empty_recording():
    with pytest.raises(ValueError, match="at least one"):
        replay_frame_index([], 0.0)


def test_dual_arm_replay_recording_round_trip(tmp_path):
    frames = [
        ReplayFrame(0.0, (0.0, 0.1, 0.2), (0.0, -0.1, -0.2), 0, 0, None, "table", "start"),
        ReplayFrame(0.5, (0.3, 0.2, 0.1), (0.3, -0.2, -0.1), 1, 1, "cycle", "both_arms", "handoff"),
    ]
    writer = object.__new__(DualArmViserRuntime)
    writer._replay_lock = threading.Lock()
    writer._replay_frames = frames
    recording = tmp_path / "handoff.json"
    writer.save_replay(recording, "Payload Handoff")

    reader = object.__new__(DualArmViserRuntime)
    reader._replay_lock = threading.Lock()
    reader._render_frame = lambda frame, replaying=False: None
    metadata = reader.load_replay(recording)
    assert metadata == {"title": "Payload Handoff", "frames": 2, "duration": 0.5}
    assert reader.replay_frames == frames


def test_dual_arm_replay_frame_rejects_invalid_payload_owner():
    with pytest.raises(ValueError, match="invalid state"):
        ReplayFrame.from_dict(
            {
                "elapsed": 0,
                "left_arm": [0, 0, 0],
                "right_arm": [0, 0, 0],
                "left_gripper": 0,
                "right_gripper": 0,
                "zone_owner": None,
                "payload_owner": "ceiling",
                "event": "start",
            }
        )


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


def test_runtime_detects_client_connected_while_registering_callback():
    class FakeServer:
        clients = {}

        def get_clients(self):
            return self.clients

        def on_client_connect(self, callback):
            # The connect event occurred before the callback became active.
            self.clients = {0: object()}
            return callback

    runtime = object.__new__(ViserDemoRuntime)
    runtime.server = FakeServer()
    runtime._monitor_stop = threading.Event()
    runtime._demo_status = type("Status", (), {"content": ""})()
    waiter = threading.Thread(target=runtime.wait_for_client)
    waiter.start()
    try:
        waiter.join(timeout=0.5)
        assert not waiter.is_alive()
        assert "client connected" in runtime._demo_status.content
    finally:
        runtime._monitor_stop.set()
        waiter.join(timeout=0.5)


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


def test_payload_attaches_only_at_pick_with_grasp_width():
    runtime = make_payload_runtime(PICK_Q, PAYLOAD_GRASP_POSITION)
    runtime.attach_payload()
    assert runtime._payload_attached
    assert not runtime._pick_object.visible
    assert runtime._carried_object.visible

    for position in (0.0, 0.2, 0.72):
        runtime = make_payload_runtime(PICK_Q, position)
        with pytest.raises(RuntimeError, match="grasp position"):
            runtime.attach_payload()


def test_payload_fits_between_fingers_throughout_grasp():
    gripper = yourdfpy.URDF.load(
        str(Path(__file__).parents[1] / "assets/Robotiq2F85/Robotiq2F85.urdf")
    )
    payload_min = PAYLOAD_OFFSET - np.asarray(PAYLOAD_DIMENSIONS) / 2
    payload_max = PAYLOAD_OFFSET + np.asarray(PAYLOAD_DIMENSIONS) / 2
    for position in np.linspace(0.0, PAYLOAD_GRASP_POSITION, 25):
        gripper.update_cfg(np.array([position]))
        for node in gripper.scene.graph.nodes_geometry:
            transform, geometry = gripper.scene.graph[node]
            mesh = gripper.scene.geometry[geometry].copy()
            mesh.apply_transform(transform)
            overlap = np.minimum(mesh.bounds[1], payload_max) - np.maximum(
                mesh.bounds[0], payload_min
            )
            assert not np.all(overlap > 1e-7), (position, node)


def test_pick_place_lifts_before_transfer_and_clears_station_surfaces():
    runtime = object.__new__(ViserDemoRuntime)
    runtime._fk_urdf = yourdfpy.URDF.load(
        str(Path(__file__).parents[1] / "assets/UR5/UR5.urdf")
    )
    for contact, above in ((PICK_Q, PICK_ABOVE_Q), (PLACE_Q, PLACE_ABOVE_Q)):
        delta = runtime.fk_position(above) - runtime.fk_position(contact)
        assert delta == pytest.approx([0.0, 0.0, 0.14], abs=1e-6)
    # Trace the actual workflow so removing a lift also invalidates this check.
    demo = Mock()
    current = HOME_Q
    carried = False
    carried_paths = []

    def move(target, steps, interval):
        nonlocal current
        if carried:
            carried_paths.append(interpolate(current, target, steps))
        current = target

    def attach():
        nonlocal carried
        carried = True

    def release():
        nonlocal carried
        carried = False

    demo.move.side_effect = move
    demo.move_and_wait.side_effect = move
    demo.attach_payload.side_effect = attach
    demo.release_payload.side_effect = release
    importlib.import_module("examples.viser.02_pick_and_place").workflow(
        demo, SimpleNamespace(steps=100, interval=0.01)
    )
    assert len(carried_paths) == 3
    assert demo.pilot.call_srv_async.call_args_list[0].args[1]["position"] == (
        PAYLOAD_GRASP_POSITION
    )
    stations = [runtime.fk_position(pose) for pose in (PICK_Q, PLACE_Q)]
    for path in carried_paths:
        for joints in path:
            center = runtime.fk_position(joints)
            half_extent = np.abs(runtime.fk_transform(joints)[:3, :3]) @ (
                np.asarray(PAYLOAD_DIMENSIONS) / 2
            )
            for station in stations:
                if np.all(np.abs(center[:2] - station[:2]) < 0.08 + half_extent[:2]):
                    bottom = center[2] - half_extent[2]
                    assert bottom >= station[2] - PAYLOAD_DIMENSIONS[2] / 2 - 1e-6
    assert runtime.fk_position(HOME_Q)[2] > max(p[2] for p in stations) + 0.14


def test_payload_world_and_tool_markers_agree_at_attachment_and_release():
    runtime = object.__new__(ViserDemoRuntime)
    runtime._fk_urdf = yourdfpy.URDF.load(
        str(Path(__file__).parents[1] / "assets/UR5/UR5.urdf")
    )
    runtime._ee_node = "/tool"
    runtime._payload_status = SimpleNamespace(content="")
    runtime.server = Mock()
    runtime.server.scene.add_box.side_effect = (
        lambda name, **kwargs: SimpleNamespace(name=name, **kwargs)
    )
    runtime.add_workspace_markers()
    for joints, marker in (
        (PICK_Q, runtime._pick_object), (PLACE_Q, runtime._placed_object)
    ):
        tool = runtime.fk_transform(joints)
        assert marker.position == pytest.approx(
            tool[:3, 3] + tool[:3, :3] @ runtime._carried_object.position
        )
        assert SO3(marker.wxyz).as_matrix() == pytest.approx(tool[:3, :3])
        station = next(
            call for call in runtime.server.scene.add_box.call_args_list
            if call.args[0] == "/world/" + ("pick" if joints is PICK_Q else "place")
        )
        top = station.kwargs["position"][2] + station.kwargs["dimensions"][2] / 2
        assert top == pytest.approx(marker.position[2] - marker.dimensions[2] / 2)


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
        return type(
            "State",
            (),
            {
                "positions": [0.2],
                "latest_sent_id": 4,
                "latest_finished_id": 4,
                "active_move_id": None,
                "cancelled_move_ids": (),
            },
        )()

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
        while (
            task.peek_response(0).result_type
            is not PeekResponseResultType.ResponseFound
        ):
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
    status = task.peek_status()
    assert status["activity"] == "idle"
    assert status["position"] == pytest.approx(0.2)
    assert status["succeeded"] == 1


def test_viser_gripper_caps_animation_update_rate():
    robot = RobotStub()
    robot.scheduler_interval = 0.001
    task = ViserGripperTask(robot)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        assert task.invoke_async(0, {"action": "close"})
        deadline = time.monotonic() + 0.5
        while task.peek_response(0).result_type is not PeekResponseResultType.ResponseFound:
            assert time.monotonic() < deadline
            time.sleep(0.005)
    _, trajectory, interval = robot.requests[0]
    assert interval == pytest.approx(1.0 / 60.0)
    assert len(trajectory) < 50


def test_viser_gripper_discovery_describes_commands_and_range():
    description = ViserGripperTask(RobotStub()).describe()
    assert description["kind"] == "actuator"
    assert description["metadata"] == {
        "device_type": "parallel_gripper",
        "commands": ["open", "close", "set_position"],
        "position_unit": "radian",
        "position_range": [0.0, 0.725],
    }


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


def test_cancel_during_gripper_preparation_prevents_motion():
    preparing, release, completed = (
        threading.Event(), threading.Event(), threading.Event()
    )

    class PreparingRobot(RobotStub):
        def query_submodule_state(self, name):
            state = super().query_submodule_state(name)
            if not preparing.is_set():
                preparing.set()
                assert release.wait(2)
            return state

    robot = PreparingRobot()
    task = ViserGripperTask(robot)
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0, {"action": "close"})
            assert preparing.wait(1)
            assert task.cancel_request(0)
            release.set()
            assert completed.wait(1)
            response = task.peek_response(0)
            assert response.result_type is PeekResponseResultType.ErrorUnknown
            assert "cancelled" in response.error.lower()
            assert robot.requests == []

            completed.clear()
            assert task.invoke_async(1, {"action": "open"})
            assert completed.wait(1)
            response = task.peek_response(1)
            assert response.result_type is PeekResponseResultType.ResponseFound
            assert [request[1][-1] for request in robot.requests] == [[0.0]]
        finally:
            release.set()
            task.stop()


@pytest.mark.parametrize("reentrant", [False, True], ids=["return-window", "callback"])
def test_cancel_before_gripper_move_id_publication_stops_motion(reentrant):
    submitted, release, completed = (
        threading.Event(), threading.Event(), threading.Event()
    )

    class PublishingRobot(RobotStub):
        def __init__(self):
            super().__init__()
            self.active_moves = set()
            self.cancelled_moves = []
            self.completed_moves = []

        def move_submodule_trajectory_async(self, name, trajectory, interval):
            super().move_submodule_trajectory_async(name, trajectory, interval)
            move_id = len(self.requests) + 3
            self.active_moves.add(move_id)
            if move_id == 4:
                if reentrant:
                    assert task._lock.acquire(timeout=0.1)
                    task._lock.release()
                    assert task.cancel_request(0)
                else:
                    submitted.set()
                    assert release.wait(2)
            return move_id

        def cancel_submodule_move(self, name, move_id):
            assert task._lock.acquire(timeout=0.1)
            task._lock.release()
            self.cancelled_moves.append(move_id)
            self.active_moves.discard(move_id)
            return True

        def wait_submodule_move(self, name, move_id, timeout):
            if move_id not in self.active_moves:
                return False
            self.active_moves.remove(move_id)
            self.completed_moves.append(move_id)
            return True

    robot = PublishingRobot()
    task = ViserGripperTask(robot)
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0, {"action": "close"})
            if not reentrant:
                assert submitted.wait(1)
                assert task.cancel_request(0)
                release.set()
            assert completed.wait(1)
            response = task.peek_response(0)
            assert response.result_type is PeekResponseResultType.ErrorUnknown
            assert "cancelled" in response.error.lower()
            assert robot.cancelled_moves == [4]
            assert robot.completed_moves == []
            assert not robot.active_moves

            completed.clear()
            assert task.invoke_async(1, {"action": "open"})
            assert completed.wait(1)
            response = task.peek_response(1)
            assert response.result_type is PeekResponseResultType.ResponseFound
            assert robot.completed_moves == [5]
        finally:
            release.set()
            task.stop()


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
    release_worker = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    blocker = executor.submit(release_worker.wait, 2.0)
    gripper_task = ViserGripperTask(robot)
    pilot = TaskPilot(
        robot, {"gripper": gripper_task}, executor_owned=executor
    )
    pilot.initialize()
    try:
        first = pilot.call_srv_async(
            "gripper", {"position": 0.2, "speed": 100.0}, MoveSyncOption.no_sync()
        )
        cancelled = pilot.call_srv_async(
            "gripper", {"position": 0.7, "speed": 100.0}, MoveSyncOption.no_sync()
        )
        last = pilot.call_srv_async(
            "gripper", {"position": 0.1, "speed": 100.0}, MoveSyncOption.no_sync()
        )
        deadline = time.monotonic() + 0.5
        while (
            gripper_task.peek_response(cancelled).result_type
            is PeekResponseResultType.ErrorRequestNotSent
        ):
            assert time.monotonic() < deadline
            time.sleep(0.002)
        # The adapter accepted this request, and its sole worker is still gated.
        assert (
            gripper_task.peek_response(cancelled).result_type
            is PeekResponseResultType.RequestSentNoResponse
        )
        assert not blocker.done()
        assert pilot.cancel_request(cancelled)
        release_worker.set()
        assert blocker.result(timeout=0.5)
        assert pilot.wait_request(first, timeout=1.0).status is RequestStatus.SUCCEEDED
        assert pilot.wait_request(last, timeout=1.0).status is RequestStatus.SUCCEEDED
        assert pilot.query_request(cancelled).status is RequestStatus.CANCELLED
        assert robot.query_submodule_state("gripper").positions == [pytest.approx(0.1)]
        assert [trajectory[-1][0] for trajectory in robot.submodule_trajectories] == [
            pytest.approx(0.2), pytest.approx(0.1)
        ]
    finally:
        release_worker.set()
        pilot.stop()
