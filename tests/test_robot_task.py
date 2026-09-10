"""Shared motion contracts and lifecycle races for the built-in robot backends."""

import concurrent.futures
import threading

import pytest

from orchestrion import MoveSyncOption, RequestStatus, SimulatedRobotTask, TaskPilot
from orchestrion.tasks import InPlaceFunctionCallTask
from orchestrion.tasks.modular_reduced_robot_task import (
    ModularReducedRobotTask,
    SubModuleTask,
)


class ModularRobot(ModularReducedRobotTask):
    def __init__(self, main=None, submodule=None):
        super().__init__(
            [0.0, 0.0],
            main or SubModuleTask("arm", 0, 1),
            [submodule or SubModuleTask("gripper", 1, 1)],
            interval=0.001,
        )

    def _on_initial_state_bg_thread(self, jps):
        pass

    def _on_robot_state_bg_thread(self, jps, jps_move_id):
        pass


@pytest.fixture(params=[SimulatedRobotTask, ModularRobot])
def robot(request):
    instance = (
        request.param([0.0]) if request.param is SimulatedRobotTask else request.param()
    )
    try:
        yield instance
    finally:
        instance.stop()


class UncopyableTrajectory(list):
    def __deepcopy__(self, memo):
        raise RuntimeError("trajectory copy failed")


def test_copy_failure_does_not_consume_main_move_id(robot):
    with pytest.raises(RuntimeError, match="copy failed"):
        robot.move_joint_trajectory_async(UncopyableTrajectory([[1.0]]))
    assert robot.query_state().latest_sent_id == -1
    assert robot.move_joint_trajectory_async([[2.0]]) == 0


def test_copy_failure_does_not_consume_submodule_move_id():
    robot = ModularRobot()
    with pytest.raises(RuntimeError, match="copy failed"):
        robot.move_submodule_trajectory_async("gripper", UncopyableTrajectory([[1.0]]))
    assert robot.query_submodule_state("gripper").latest_sent_id == -1
    assert robot.move_submodule_trajectory_async("gripper", [[2.0]]) == 0
    robot.stop()


def test_main_move_cancellation_survives_restart(robot):
    cancelled = robot.move_joint_trajectory_async([[1.0]])
    robot.stop()
    assert cancelled in robot.query_state().cancelled_move_ids
    robot.initialize()
    assert not robot.wait_move(time_out=0.0)
    next_move = robot.move_joint_trajectory_async([[2.0]], interval=0.001)
    assert next_move > cancelled
    assert robot.wait_move(time_out=1.0)
    assert cancelled in robot.query_state().cancelled_move_ids


def test_restart_preserves_completed_main_move_id(robot):
    robot.initialize()
    completed = robot.move_joint_trajectory_async([[1.0]], interval=0.001)
    assert robot.wait_move(time_out=1.0)
    robot.stop()
    robot.initialize()
    assert robot.wait_move(time_out=0.1)
    assert robot.query_state().latest_finished_id == completed


def test_main_sync_is_cancelled_when_robot_stops_independently(robot):
    class EchoTask(InPlaceFunctionCallTask):
        def _call_fn(self, request_id, content):
            return content

    with TaskPilot(robot, {"echo": EchoTask()}, poll_interval=5.0) as pilot:
        move_id = robot.move_joint_trajectory_async([[1.0], [2.0]], interval=10.0)
        request_id = pilot.call_srv_async(
            "echo", sync_option=MoveSyncOption.sync_w_explicit_id(move_id)
        )
        robot.stop()
        result = pilot.wait_request(request_id, timeout=1.0)
        assert result.status is RequestStatus.CANCELLED
        assert "move was cancelled" in result.error


def test_simulated_move_id_allocation_and_enqueue_are_atomic():
    robot = SimulatedRobotTask([0.0])
    first_copying = threading.Event()
    release = threading.Event()

    class PausedTrajectory(list):
        def __deepcopy__(self, memo):
            first_copying.set()
            assert release.wait(2.0)
            return [[1.0]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            robot.move_joint_trajectory_async, PausedTrajectory([[1.0]])
        )
        try:
            assert first_copying.wait(1.0)
            second = executor.submit(robot.move_joint_trajectory_async, [[2.0]])
        finally:
            release.set()
        first_id, second_id = first.result(1.0), second.result(1.0)
    queued = [robot._queue.get_nowait() for _ in range(2)]
    assert [item.motion_id_begin for item in queued] == sorted([first_id, second_id])
    robot.stop()


@pytest.mark.parametrize(
    "trajectory", [None, 1.0, {}, [None], [1.0], ["x"], [[1.0], 2.0]]
)
def test_malformed_trajectory_is_rejected_without_consuming_ids(robot, trajectory):
    assert robot.move_joint_trajectory_async(trajectory) == -1
    assert robot.query_state().latest_sent_id == -1


def test_trajectory_prepared_before_stop_cannot_enter_restarted_robot(robot):
    copying = threading.Event()
    release = threading.Event()

    class PausedTrajectory(list):
        def __deepcopy__(self, memo):
            copying.set()
            assert release.wait(2.0)
            return [[1.0]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        submitted = executor.submit(
            robot.move_joint_trajectory_async, PausedTrajectory([[1.0]])
        )
        try:
            assert copying.wait(1.0)
            robot.stop()
            robot.initialize()
        finally:
            release.set()
        assert submitted.result(1.0) == -1
    assert robot.query_state().latest_sent_id == -1


def test_stopped_robot_rejects_new_motion_until_reinitialized(robot):
    robot.stop()
    assert robot.move_joint_trajectory_async([[1.0]]) == -1
    if isinstance(robot, ModularRobot):
        assert robot.move_submodule_trajectory_async("gripper", [[1.0]]) == -1
        assert robot.move_submodules_trajectories_async({"gripper": [[1.0]]}) == {}
    assert robot.query_state().latest_sent_id == -1


def test_cancelled_submodule_wait_cannot_succeed_when_a_later_move_finishes(
    monkeypatch,
):
    robot = ModularRobot()
    cancelled = robot.move_submodule_trajectory_async("gripper", [[1.0]])
    later = robot.move_submodule_trajectory_async("gripper", [[2.0]])
    waiting = threading.Event()
    original_wait = robot._state_condition.wait

    def record_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(robot._state_condition, "wait", record_wait)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(robot.wait_submodule_move, "gripper", cancelled, 1.0)
        assert waiting.wait(1.0)
        with robot._state_condition:
            robot.cancel_submodule_move("gripper", cancelled)
            robot._submodule_finished_ids["gripper"] = later
            robot._state_condition.notify_all()
        assert result.result(1.0) is False
    robot.stop()


def test_module_layout_is_isolated_from_later_configuration_mutation():
    main = SubModuleTask("arm", 0, 1)
    submodule = SubModuleTask("gripper", 1, 1)
    robot = ModularRobot(main, submodule)
    main.n_dof = 99
    submodule.dof_begin = 0
    assert robot.move_joint_trajectory_async([[1.0]]) == 0
    robot.initialize()
    try:
        move_id = robot.move_submodule_trajectory_async("gripper", [[2.0]])
        assert robot.wait_submodule_move("gripper", move_id, timeout=1.0)
        assert robot.wait_move(time_out=1.0)
        assert robot.query_state().jps == [1.0, 2.0]
    finally:
        robot.stop()


def test_tuple_trajectory_and_endpoints_execute_without_worker_failure(robot):
    robot.initialize()
    assert (
        robot.move_joint_trajectory_async(
            ((1.0,), (2.0,)), interval=0.001, endpoint_index=(1, 2)
        )
        == 0
    )
    assert robot.wait_move(time_out=1.0)
    assert robot.query_state().jps[0] == 2.0


@pytest.mark.parametrize("endpoints", [1, True, "1", {1}])
def test_non_sequence_endpoint_container_is_rejected(robot, endpoints):
    assert robot.move_joint_trajectory_async([[1.0]], endpoint_index=endpoints) == -1


def test_state_observer_can_stop_motion_and_discard_queued_work(robot):
    requested = threading.Event()
    returned = threading.Event()

    def stop_after_first_point():
        if robot.query_state().jps[0] >= 1.0 and not requested.is_set():
            requested.set()
            robot.stop()
            returned.set()

    robot.set_state_change_callback(stop_after_first_point)
    robot.move_joint_trajectory_async([[1.0], [2.0]], interval=0.001)
    robot.move_joint_trajectory_async([[9.0]], interval=0.001)
    robot.initialize()
    assert returned.wait(1.0)
    robot.stop()
    robot.initialize()
    assert robot.query_state().jps[0] == 1.0
    assert robot.query_state().cancelled_move_ids == (0, 1)


def test_failed_worker_start_can_be_stopped_and_retried(robot, monkeypatch):
    original_start = threading.Thread.start

    def fail_start(thread):
        raise RuntimeError("worker could not start")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="could not start"):
        robot.initialize()
    robot.stop()
    monkeypatch.setattr(threading.Thread, "start", original_start)
    robot.initialize()
    assert robot.move_joint_trajectory_async([[1.0]], interval=0.001) >= 0
    assert robot.wait_move(time_out=1.0)
