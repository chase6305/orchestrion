import concurrent.futures
import importlib
import threading
import time
from collections import deque
from contextlib import nullcontext
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
from orchestrion.tasks.modular_reduced_robot_task import (
    SubModuleMovementRequest,
    _SubmoduleSchedule,
)
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


def test_viser_startup_advertises_the_actual_bound_port(monkeypatch, capsys):
    runtime = SimpleNamespace(
        server=SimpleNamespace(get_port=lambda: 8081),
        set_status=Mock(),
        hold=Mock(),
    )
    factory = Mock(return_value=nullcontext(runtime))
    module = importlib.import_module("examples.viser.common")
    monkeypatch.setattr(module, "ViserDemoRuntime", factory)
    module.run_demo(
        "demo", Mock(),
        argv=["--port", "8080", "--duration", "0", "--startup-delay", "0"],
    )
    output = capsys.readouterr().out
    assert "Open http://localhost:8081" in output
    assert "ssh -L 8081:localhost:8081" in output
    assert factory.call_args.kwargs["port"] == 8080


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
        for value in (0.0, 0.2, 0.2, 0.7)
    ]
    assert replay_frame_index(frames, -1.0) == 0
    assert replay_frame_index(frames, 0.19) == 0
    assert replay_frame_index(frames, 0.2) == 2
    assert replay_frame_index(frames, 99.0) == 3


def test_replay_cursor_uses_bounded_indexed_reads_for_long_recordings():
    class IndexedFrames:
        reads = 0

        def __len__(self):
            return 1000000

        def __getitem__(self, index):
            assert 0 <= index < len(self)
            self.reads += 1
            return SimpleNamespace(elapsed=index * 0.01)

        def __iter__(self):
            raise AssertionError("replay lookup must not scan every frame")

    frames = IndexedFrames()
    assert replay_frame_index(frames, 1234.565) == 123456
    assert frames.reads <= 21


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


@pytest.mark.parametrize("owners", [[], ["left_arm"], ["left_arm", "right_arm"]])
def test_replay_capture_uses_one_event_snapshot_for_stage_and_payload(owners):
    events = [
        {"event": "payload_owner_changed", "owner": owner} for owner in owners
    ]
    events.append({"event": "snapshot-stage"})

    class Runtime(DualArmViserRuntime):
        event_reads = 0

        @property
        def events(self):
            self.event_reads += 1
            if self.event_reads == 1:
                return events
            return [{"event": "payload_owner_changed", "owner": "output_table"}]

    runtime = object.__new__(Runtime)
    runtime._capture_started_at = None
    runtime._replay_lock = threading.Lock()
    runtime._replay_frames = []
    frame = runtime._capture_frame(
        {"left": [0.0] * 3, "right": [0.0] * 3},
        {"left": 0.0, "right": 0.0},
        None,
    )
    assert frame.payload_owner == (owners[-1] if owners else "table")
    assert frame.event == "snapshot-stage"
    assert runtime.event_reads == 1
    assert runtime.replay_frames == [frame]


def test_concurrent_replay_capture_keeps_timestamps_in_append_order(monkeypatch):
    reading, release = threading.Event(), threading.Event()
    event_lock = threading.Lock()

    class Runtime(DualArmViserRuntime):
        event_reads = 0

        @property
        def events(self):
            with event_lock:
                self.event_reads += 1
                first_read = self.event_reads == 1
            if first_read:
                reading.set()
                assert release.wait(2)
            return []

    clock = iter([10.0, 20.0])
    module = importlib.import_module("examples.dual_arm.viser")
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    runtime = object.__new__(Runtime)
    runtime._capture_started_at = None
    runtime._replay_lock = threading.Lock()
    runtime._replay_frames = []
    arguments = (
        {"left": [0.0] * 3, "right": [0.0] * 3},
        {"left": 0.0, "right": 0.0},
        None,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(runtime._capture_frame, *arguments)
        try:
            assert reading.wait(1)
            executor.submit(runtime._capture_frame, *arguments).result(timeout=1)
        finally:
            release.set()
        first.result(timeout=1)
    frames = runtime.replay_frames
    assert frames[0].elapsed == 0.0
    assert frames[1].elapsed >= frames[0].elapsed


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


def test_viser_interpolation_keeps_finite_endpoints_and_independent_points():
    start = np.array([1e308, -1e308])
    end = -start
    trajectory = interpolate(start, end, 5)
    assert trajectory[0] == start.tolist()
    assert trajectory[-1] == end.tolist()
    assert trajectory[2] == [0.0, 0.0]
    assert np.isfinite(trajectory).all()
    trajectory[0][0] = 0.0
    assert start[0] == 1e308
    assert trajectory[1][0] != 0.0


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


@pytest.mark.parametrize("operation", ["invoke_async", "cancel_request"])
@pytest.mark.parametrize(
    "request_id, error_type",
    [(True, TypeError), (1.0, TypeError), (-1, ValueError), ("1", TypeError)],
)
def test_gripper_rejects_invalid_ids_before_changing_active_motion(
    monkeypatch, operation, request_id, error_type,
):
    robot = RobotStub()
    entered, release = threading.Event(), threading.Event()
    cancel_move = Mock(wraps=robot.cancel_submodule_move)
    monkeypatch.setattr(robot, "cancel_submodule_move", cancel_move)

    def wait_for_motion(name, move_id, timeout):
        entered.set()
        assert release.wait(2)
        return True

    monkeypatch.setattr(robot, "wait_submodule_move", wait_for_motion)
    task = ViserGripperTask(robot)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(1, {"action": "close"})
            assert entered.wait(1)
            with pytest.raises(error_type, match="request_id"):
                if operation == "invoke_async":
                    task.invoke_async(request_id, {"action": "open"})
                else:
                    task.cancel_request(request_id)
            cancel_move.assert_not_called()
            assert len(robot.requests) == 1
        finally:
            release.set()
            task._future_map[1].result(timeout=1)
            task.stop()
    assert task.peek_response(1).result_type is PeekResponseResultType.ResponseFound


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


@pytest.mark.parametrize(
    "current, target",
    [(0.2, 0.2), (0.0, 0.0), (0.0, 5e-324)],
    ids=["stationary", "stationary-zero", "subnormal-move"],
)
def test_viser_gripper_handles_tiny_speeds_without_interval_underflow(current, target):
    class Robot(RobotStub):
        def query_submodule_state(self, name):
            return SimpleNamespace(positions=[current])

    robot = Robot()
    task = ViserGripperTask(robot)
    completed = threading.Event()
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0, {"position": target, "speed": 5e-324})
            assert completed.wait(1)
            response = task.peek_response(0)
            assert response.result_type is PeekResponseResultType.ResponseFound
        finally:
            task.stop()
    _, trajectory, interval = robot.requests[0]
    assert trajectory[0] == [current]
    assert trajectory[-1] == [target]
    assert (len(trajectory) - 1) * interval == pytest.approx(
        max(interval, abs(target - current) / 5e-324)
    )


def test_slow_gripper_motion_bounds_precomputed_points_and_preserves_duration():
    robot = RobotStub()
    task = ViserGripperTask(robot)
    task._execute_call(0, {"position": 0.72, "speed": 0.00065})
    _, trajectory, interval = robot.requests[0]
    assert len(trajectory) <= 4096
    assert trajectory[0] == [0.2]
    assert trajectory[-1] == [0.72]
    assert (len(trajectory) - 1) * interval == pytest.approx(800.0)


@pytest.mark.parametrize("scheduler_interval, speed", [(0.01, 0.00005), (3.0, 1.0)])
def test_gripper_default_timeout_covers_the_final_sample_hold(
    scheduler_interval, speed,
):
    class TimedRobot(RobotStub):
        def wait_submodule_move(self, name, move_id, timeout):
            _, trajectory, interval = self.requests[-1]
            return timeout > len(trajectory) * interval

    robot = TimedRobot()
    robot.scheduler_interval = scheduler_interval
    result = ViserGripperTask(robot)._execute_call(
        0, {"position": 0.72, "speed": speed}
    )
    assert result["position"] == 0.72


@pytest.mark.parametrize("speed", [0.02, 0.005])
def test_gripper_default_timeout_covers_scheduler_tick_rounding(monkeypatch, speed):
    worker = ModularRobotStub(
        [0.0, 0.2], SubModuleTask("arm", 0, 1),
        [SubModuleTask("gripper", 1, 1)], interval=0.01,
    )
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        "orchestrion.tasks.modular_reduced_robot_task.time",
        SimpleNamespace(monotonic=lambda: clock.now),
    )

    class TimedRobot(RobotStub):
        def wait_submodule_move(self, name, move_id, timeout):
            _, trajectory, interval = self.requests[-1]
            schedule = _SubmoduleSchedule(deque([
                SubModuleMovementRequest(name, trajectory, interval, move_id),
            ]))
            # Run production scheduling on virtual ticks, without a wall-clock wait.
            while clock.now < timeout:
                worker._advance_submodule_moves({name: schedule}, [0.0, 0.2])
                if not schedule.moves:
                    return True
                clock.now += self.scheduler_interval
            return False

    result = ViserGripperTask(TimedRobot())._execute_call(
        0, {"action": "close", "speed": speed}
    )
    assert result["position"] == 0.72


def test_unrepresentable_gripper_motion_duration_fails_before_enqueueing():
    robot = RobotStub()
    task = ViserGripperTask(robot)
    with pytest.raises(ValueError, match="duration must be finite"):
        task._execute_call(0, {"position": 0.72, "speed": 5e-324, "timeout": 1})
    assert robot.requests == []


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


def test_concurrent_gripper_cancellations_release_queued_sequence_once():
    rendezvous = threading.Barrier(2)

    class QueuedFuture(concurrent.futures.Future):
        def cancel(self):
            rendezvous.wait(timeout=1)
            return super().cancel()

    future = QueuedFuture()
    executor = SimpleNamespace(submit=lambda *args: future)
    robot = RobotStub()
    task = ViserGripperTask(robot)
    task.initialize(executor=executor)
    try:
        assert task.invoke_async(0, {"action": "close"})
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as callers:
            cancellations = [callers.submit(task.cancel_request, 0) for _ in range(2)]
            assert all(cancellation.result(timeout=2) for cancellation in cancellations)
        assert not robot.requests
        assert not task._request_sequences
        assert not task._cancelled_requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as workers:
            task.initialize(executor=workers)
            assert task.invoke_async(1, {"action": "open"})
            task._future_map[1].result(timeout=1)
        assert len(robot.requests) == 1
    finally:
        concurrent.futures.Future.cancel(future)
        task.stop()


def test_gripper_completion_clears_cancellation_after_handler_returns():
    returned, release, completed = (
        threading.Event(), threading.Event(), threading.Event()
    )

    class PausedTask(ViserGripperTask):
        def _call_fn(self, request_id, content=None):
            result = super()._call_fn(request_id, content)
            returned.set()
            assert release.wait(2)
            return result

    task = PausedTask(RobotStub())
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0, {"action": "close"})
            assert returned.wait(1)
            assert task.cancel_request(0)
            release.set()
            assert completed.wait(1)
            assert not task._cancelled_requests
            assert not task._request_sequences
        finally:
            release.set()
            task.stop()


def test_stopping_gripper_ignores_completed_history_and_cancels_queued_work(
    monkeypatch,
):
    robot = RobotStub()
    task = ViserGripperTask(robot)
    completed = threading.Event()
    release_worker = threading.Event()
    task.set_completion_callback(completed.set)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0, {"action": "close"})
            assert completed.wait(1)
            finished = task._future_map[0]
            inspect_finished = Mock(wraps=finished.done)
            monkeypatch.setattr(finished, "done", inspect_finished)
            blocker = executor.submit(release_worker.wait, 2.0)
            assert task.invoke_async(1, {"action": "open"})
            task.stop()
            inspect_finished.assert_not_called()
        finally:
            release_worker.set()
            task.stop()
        blocker.result(timeout=1)
    assert len(robot.requests) == 1
    assert task.peek_response(0).result_type is PeekResponseResultType.ResponseFound
    assert task.peek_response(1).result_type is PeekResponseResultType.ErrorUnknown


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
