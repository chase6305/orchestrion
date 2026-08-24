import threading
import time

import pytest

from orchestrion.tasks.modular_reduced_robot_task import (
    ModularReducedRobotTask,
    SubModuleTask,
)


class RobotTaskStub(ModularReducedRobotTask):
    def _on_robot_state_bg_thread(self, jps, jps_move_id):
        pass

    def _on_initial_state_bg_thread(self, jps):
        pass


class RecordingRobotTaskStub(RobotTaskStub):
    def __init__(self, *args, **kwargs):
        self.updates = []
        super().__init__(*args, **kwargs)

    def _on_robot_state_bg_thread(self, jps, jps_move_id):
        self.updates.append(jps.copy())


def make_robot(interval=0.005):
    main = SubModuleTask(name="arm", dof_begin=0, n_dof=3)
    gripper = SubModuleTask(name="gripper", dof_begin=3, n_dof=1)
    return RobotTaskStub([0.0] * 4, main, [gripper], interval=interval)


@pytest.mark.parametrize("interval", [0.0, -0.01, float("nan"), float("inf")])
def test_scheduler_rejects_invalid_interval(interval):
    with pytest.raises(ValueError, match="interval"):
        make_robot(interval)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.0", True])
def test_robot_rejects_invalid_initial_joints(value):
    with pytest.raises(ValueError, match="Initial joints"):
        RobotTaskStub(
            [value, 0.0],
            SubModuleTask("arm", 0, 1),
            [SubModuleTask("gripper", 1, 1)],
        )


def test_module_layout_rejects_duplicate_name_and_overlapping_dofs():
    main = SubModuleTask(name="arm", dof_begin=0, n_dof=3)
    with pytest.raises(ValueError, match="names"):
        RobotTaskStub(
            [0.0] * 4,
            main,
            [SubModuleTask(name="arm", dof_begin=3, n_dof=1)],
        )
    with pytest.raises(ValueError, match="overlap"):
        RobotTaskStub(
            [0.0] * 4,
            main,
            [SubModuleTask(name="gripper", dof_begin=2, n_dof=1)],
        )


@pytest.mark.parametrize("name", ["", 1, None])
def test_module_layout_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="name"):
        RobotTaskStub([0.0], SubModuleTask(name, 0, 1), [])


@pytest.mark.parametrize(
    "dof_begin,n_dof", [(0.0, 1), (0, 1.0), (False, 1), (0, True)]
)
def test_module_layout_rejects_non_integer_dof_ranges(dof_begin, n_dof):
    with pytest.raises(TypeError, match="integers"):
        RobotTaskStub([0.0], SubModuleTask("arm", dof_begin, n_dof), [])


@pytest.mark.parametrize("interval", [0.0, -0.01, float("nan"), float("inf")])
def test_motion_rejects_invalid_interval(interval):
    robot = make_robot()
    assert robot.move_joint_trajectory_async([[0.0, 0.0, 0.0]], interval) == -1
    assert robot.move_submodule_trajectory_async("gripper", [[0.0]], interval) == -1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.1", True])
def test_motion_rejects_non_finite_or_non_numeric_joints(value):
    robot = make_robot()
    assert robot.move_joint_trajectory_async([[value, 0.0, 0.0]]) == -1
    assert robot.move_submodule_trajectory_async("gripper", [[value]]) == -1


@pytest.mark.parametrize("endpoints", [[], [1.0], [True]])
def test_main_motion_rejects_invalid_endpoint_types(endpoints):
    robot = make_robot()
    assert (
        robot.move_joint_trajectory_async(
            [[0.0, 0.0, 0.0]], endpoint_index=endpoints
        )
        == -1
    )


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
def test_wait_move_rejects_invalid_timing(time_out, interval):
    with pytest.raises(ValueError):
        make_robot().wait_move(time_out=time_out, interval=interval)


@pytest.mark.parametrize("timeout", [-0.1, float("nan"), float("inf")])
def test_robot_stop_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        make_robot().stop(timeout=timeout)


def test_wait_move_timeout_is_not_extended_by_poll_interval():
    robot = make_robot()
    robot.move_joint_trajectory_async([[0.1, 0.0, 0.0]])
    started = time.monotonic()
    assert not robot.wait_move(time_out=0.02, interval=1.0)
    assert time.monotonic() - started < 0.1


def test_wait_move_is_woken_by_completion_before_poll_interval():
    robot = make_robot(interval=0.002)
    robot.initialize()
    try:
        robot.move_joint_trajectory_async([[0.1, 0.0, 0.0]])
        started = time.monotonic()
        assert robot.wait_move(time_out=0.5, interval=1.0)
        assert time.monotonic() - started < 0.1
    finally:
        robot.stop()


def wait_until(predicate, timeout=0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_main_trajectory_updates_state_and_move_ids():
    robot = make_robot()
    robot.initialize()
    try:
        begin = robot.move_joint_trajectory_async(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], endpoint_index=[1, 2]
        )
        assert begin == 0
        assert robot.wait_move(time_out=0.5)
        state = robot.query_state()
        assert state.latest_sent_id == 1
        assert state.latest_finished_id == 1
        assert state.jps == [0.4, 0.5, 0.6, 0.0]
    finally:
        robot.stop()


def test_submodule_trajectory_updates_only_its_joints():
    robot = make_robot()
    robot.initialize()
    try:
        assert robot.move_submodule_async("gripper", [[0.4], [0.8]])
        assert wait_until(lambda: robot.query_state().jps[3] == 0.8)
        assert robot.query_state().jps[:3] == [0.0, 0.0, 0.0]
    finally:
        robot.stop()


def test_invalid_trajectories_are_rejected():
    robot = make_robot()
    assert robot.move_joint_trajectory_async([]) == -1
    assert robot.move_joint_trajectory_async([[1.0, 2.0]]) == -1
    assert (
        robot.move_joint_trajectory_async(
            [[1.0, 2.0, 3.0]], endpoint_index=[0]
        )
        == -1
    )
    assert not robot.move_submodule_async("missing", [[1.0]])
    assert not robot.move_submodule_async("gripper", [[1.0, 2.0]])


def test_query_state_returns_a_copy():
    robot = make_robot()
    state = robot.query_state()
    state.jps[0] = 99.0
    assert robot.query_state().jps[0] == 0.0


def test_robot_can_restart():
    robot = make_robot()
    robot.initialize()
    robot.stop()
    robot.initialize()
    robot.stop()


def test_restart_does_not_execute_main_motion_queued_before_stop():
    robot = make_robot(interval=0.002)
    robot.initialize()
    robot.move_joint_trajectory_async(
        [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], interval=0.1
    )
    robot.move_joint_trajectory_async([[0.9, 0.0, 0.0]], interval=0.002)
    time.sleep(0.01)
    robot.stop()
    stopped_position = robot.query_state().jps.copy()
    robot.initialize()
    try:
        time.sleep(0.05)
        assert robot.query_state().jps == stopped_position
    finally:
        robot.stop()


def test_submodule_move_has_state_and_completion_id():
    robot = make_robot(interval=0.002)
    robot.initialize()
    try:
        move_id = robot.move_submodule_trajectory_async(
            "gripper", [[0.1], [0.2], [0.3]]
        )
        assert move_id == 0
        assert robot.query_submodule_state("gripper").latest_sent_id == 0
        assert robot.wait_submodule_move("gripper", move_id, timeout=0.5)
        state = robot.query_submodule_state("gripper")
        assert state.positions == [0.3]
        assert state.latest_finished_id == 0
    finally:
        robot.stop()


def test_submodule_move_can_be_cancelled():
    robot = make_robot(interval=0.01)
    robot.initialize()
    try:
        move_id = robot.move_submodule_trajectory_async(
            "gripper", [[value / 100.0] for value in range(50)]
        )
        assert robot.cancel_submodule_move("gripper", move_id)
        assert not robot.wait_submodule_move("gripper", move_id, timeout=0.5)
        state = robot.query_submodule_state("gripper")
        assert state.active_move_id is None
        assert state.cancelled_move_ids == (move_id,)
        assert not robot.cancel_submodule_move("gripper", move_id + 1)
        next_move = robot.move_submodule_trajectory_async("gripper", [[0.2]])
        assert robot.wait_submodule_move("gripper", next_move, timeout=0.5)
        assert not robot.wait_submodule_move("gripper", move_id, timeout=0.01)
    finally:
        robot.stop()


def test_submodule_respects_slower_request_interval():
    robot = make_robot(interval=0.002)
    robot.initialize()
    try:
        started = time.monotonic()
        move_id = robot.move_submodule_trajectory_async(
            "gripper", [[0.1], [0.2], [0.3]], interval=0.03
        )
        assert robot.wait_submodule_move("gripper", move_id, timeout=0.5)
        assert time.monotonic() - started >= 0.055
    finally:
        robot.stop()


def test_main_motion_respects_slower_request_interval():
    robot = make_robot(interval=0.002)
    robot.initialize()
    try:
        started = time.monotonic()
        robot.move_joint_trajectory_async(
            [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
            interval=0.03,
        )
        assert robot.wait_move(time_out=0.5)
        assert time.monotonic() - started >= 0.055
    finally:
        robot.stop()


def test_submodule_callbacks_continue_while_main_waits_for_next_step():
    robot = RecordingRobotTaskStub(
        [0.0] * 4,
        SubModuleTask(name="arm", dof_begin=0, n_dof=3),
        [SubModuleTask(name="gripper", dof_begin=3, n_dof=1)],
        interval=0.002,
    )
    robot.initialize()
    try:
        robot.move_joint_trajectory_async(
            [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], interval=0.1
        )
        move_id = robot.move_submodule_trajectory_async(
            "gripper", [[0.4], [0.6]], interval=0.002
        )
        assert robot.wait_submodule_move("gripper", move_id, timeout=0.2)
        assert any(update[-1] == 0.6 for update in robot.updates)
        assert not robot.wait_move(time_out=0.01)
    finally:
        robot.stop()


def test_state_change_callback_is_notified_by_main_and_submodule_motion():
    robot = make_robot(interval=0.002)
    notified = threading.Event()
    robot.set_state_change_callback(notified.set)
    robot.initialize()
    try:
        robot.move_joint_trajectory_async([[0.1, 0.0, 0.0]])
        assert notified.wait(0.2)
        notified.clear()
        robot.move_submodule_trajectory_async("gripper", [[0.5]])
        assert notified.wait(0.2)
    finally:
        robot.stop()


def test_stopping_robot_cancels_all_unfinished_submodule_moves():
    robot = make_robot(interval=0.005)
    robot.initialize()
    move_ids = [
        robot.move_submodule_trajectory_async(
            "gripper", [[value / 100.0] for value in range(50)]
        )
        for _ in range(3)
    ]
    robot.stop()
    state = robot.query_submodule_state("gripper")
    assert state.active_move_id is None
    assert set(move_ids).issubset(state.cancelled_move_ids)
    assert all(
        not robot.wait_submodule_move("gripper", move_id, timeout=0.01)
        for move_id in move_ids
    )
