"""Shared motion contracts and lifecycle races for the built-in robot backends."""

import concurrent.futures
import threading
import weakref
from dataclasses import dataclass, field
from fractions import Fraction

import pytest

from orchestrion import MoveSyncOption, RequestStatus, SimulatedRobotTask, TaskPilot
from orchestrion.tasks import InPlaceFunctionCallTask
from orchestrion.tasks.modular_reduced_robot_task import (
    ModularReducedRobotTask,
    SubModuleTask,
)
from orchestrion.utils.logger import logger


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


def test_copy_failure_does_not_consume_main_move_id(robot, caplog):
    assert robot.move_joint_trajectory_async(UncopyableTrajectory([[1.0]])) == -1
    assert "trajectory copy failed" in caplog.text
    assert robot.query_state().latest_sent_id == -1
    assert robot.move_joint_trajectory_async([[2.0]]) == 0


def test_copy_failure_does_not_consume_submodule_move_id(caplog):
    robot = ModularRobot()
    assert (
        robot.move_submodule_trajectory_async("gripper", UncopyableTrajectory([[1.0]]))
        == -1
    )
    assert "trajectory copy failed" in caplog.text
    assert robot.query_submodule_state("gripper").latest_sent_id == -1
    assert robot.move_submodule_trajectory_async("gripper", [[2.0]]) == 0
    robot.stop()


def test_fraction_motion_timing_and_stop(robot):
    interval = Fraction(1, 1000)
    robot.initialize()
    try:
        assert robot.move_joint_trajectory_async([[0.5]], interval=interval) == 0
        assert robot.wait_move(time_out=Fraction(1), interval=interval)
        assert robot.query_state().jps[0] == 0.5
        robot.stop(timeout=Fraction(1))
    finally:
        robot.stop()


@pytest.mark.parametrize("interval", [Fraction(1, 1000), 1e300], ids=["fraction", "huge"])
def test_main_wait_accepts_real_intervals_and_large_deadlines(
    robot, monkeypatch, interval
):
    waiting = threading.Event()
    original_wait = robot._state_condition.wait

    def observe_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(robot._state_condition, "wait", observe_wait)
    assert robot.move_joint_trajectory_async([[0.5]]) == 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(robot.wait_move, 1e300, interval)
        try:
            assert waiting.wait(1)
            with pytest.raises(concurrent.futures.TimeoutError):
                pending.result(timeout=0.01)
        finally:
            robot.stop(timeout=1)
        assert pending.result(timeout=1) is False


@pytest.mark.parametrize("group", [False, True], ids=["single", "group"])
def test_submodule_large_deadline_is_woken_by_cancellation(monkeypatch, group):
    robot = ModularRobot()
    waiting = threading.Event()
    original_wait = robot._state_condition.wait

    def observe_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(robot._state_condition, "wait", observe_wait)
    move_id = robot.move_submodule_trajectory_async("gripper", [[0.5]])
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = (
            executor.submit(robot.wait_submodule_moves, {"gripper": move_id}, 1e300)
            if group
            else executor.submit(robot.wait_submodule_move, "gripper", move_id, 1e300)
        )
        try:
            assert waiting.wait(1)
            with pytest.raises(concurrent.futures.TimeoutError):
                pending.result(timeout=0.01)
        finally:
            robot.stop(timeout=1)
        assert pending.result(timeout=1) is False


def test_worker_failure_cancels_queued_motion_and_wakes_waiters(robot, monkeypatch):
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    step_name = (
        "_run_next_trajectory"
        if isinstance(robot, SimulatedRobotTask)
        else "_process_motion"
    )
    original_step = getattr(robot, step_name)
    original_wait = robot._state_condition.wait

    class BrokenMessageError(RuntimeError):
        def __str__(self):
            raise ValueError("broken device exception formatter")

    def fail_step():
        entered.set()
        assert release.wait(1.0)
        raise BrokenMessageError()

    def observe_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(robot, step_name, fail_step)
    monkeypatch.setattr(robot._state_condition, "wait", observe_wait)
    first = robot.move_joint_trajectory_async([[1.0]])
    second = robot.move_joint_trajectory_async([[2.0]])
    robot.initialize()
    try:
        assert entered.wait(1.0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(robot.wait_move)
            try:
                assert waiting.wait(1.0)
                release.set()
                assert result.result(timeout=1.0) is False
            finally:
                release.set()
                robot.stop(timeout=1.0)
        assert {first, second}.issubset(robot.query_state().cancelled_move_ids)
        assert robot.move_joint_trajectory_async([[3.0]]) == -1

        monkeypatch.setattr(robot, step_name, original_step)
        robot.initialize()
        later = robot.move_joint_trajectory_async([[4.0]], interval=0.001)
        assert later > second
        assert robot.wait_move(time_out=1.0)
        assert {first, second}.issubset(robot.query_state().cancelled_move_ids)
    finally:
        release.set()
        robot.stop(timeout=1.0)


@pytest.mark.parametrize("observer_fails", [False, True])
def test_worker_logging_failure_discards_queued_motion_before_restart(
    robot, monkeypatch, observer_fails
):
    simulated = isinstance(robot, SimulatedRobotTask)
    step_name = "_run_next_trajectory" if simulated else "_process_motion"
    worker = robot._run if simulated else robot._bg_thread_loop

    def fail_step():
        raise ValueError("device failure")

    def fail_log(*args, **kwargs):
        raise RuntimeError("log sink unavailable")

    def fail_observer():
        raise ValueError("state observer failed")

    first = robot.move_joint_trajectory_async([[1.0]], interval=0.001)
    second = robot.move_joint_trajectory_async([[2.0]], interval=0.001)
    if not simulated:
        submodule_move = robot.move_submodule_trajectory_async(
            "gripper", [[3.0]], interval=0.001
        )
    if observer_fails:
        robot.set_state_change_callback(fail_observer)
    with monkeypatch.context() as failure:
        failure.setattr(robot, step_name, fail_step)
        failure.setattr(logger, "error", fail_log)
        assert worker() is None
    assert {first, second}.issubset(robot.query_state().cancelled_move_ids)
    assert robot.move_joint_trajectory_async([[4.0]]) == -1
    if not simulated:
        assert submodule_move in robot.query_submodule_state("gripper").cancelled_move_ids

    observed = []
    robot.set_state_change_callback(lambda: observed.append(robot.query_state().jps))
    robot.initialize()
    try:
        later = robot.move_joint_trajectory_async([[4.0]], interval=0.001)
        assert later > second
        assert robot.wait_move(time_out=1.0)
        assert all(joints[0] not in {1.0, 2.0} for joints in observed)
        if not simulated:
            assert robot.query_submodule_state("gripper").positions == [0.0]
    finally:
        robot.stop(timeout=1.0)


def test_timed_out_stop_discards_queued_motion_before_restart(robot):
    entered, release = threading.Event(), threading.Event()
    simulated = isinstance(robot, SimulatedRobotTask)

    def block_after_first_point():
        if robot.query_state().jps[0] == 1.0 and not entered.is_set():
            entered.set()
            assert release.wait(2.0)

    robot.set_state_change_callback(block_after_first_point)
    robot.move_joint_trajectory_async([[1.0]], interval=0.001)
    robot.initialize()
    worker = robot._thread if simulated else robot._bg_thread
    try:
        assert entered.wait(1.0)
        cancelled = robot.move_joint_trajectory_async([[2.0]], interval=0.001)
        if not simulated:
            submodule_move = robot.move_submodule_trajectory_async(
                "gripper", [[3.0]], interval=0.001
            )
        with pytest.raises(TimeoutError):
            robot.stop(timeout=0.0)
        assert cancelled in robot.query_state().cancelled_move_ids
        release.set()
        worker.join(timeout=1.0)
        assert not worker.is_alive()

        observed = []
        robot.set_state_change_callback(lambda: observed.append(robot.query_state().jps))
        robot.initialize()
        assert robot.move_joint_trajectory_async([[4.0]], interval=0.001) > cancelled
        assert robot.wait_move(time_out=1.0)
        assert all(joints[0] != 2.0 for joints in observed)
        if not simulated:
            state = robot.query_submodule_state("gripper")
            assert submodule_move in state.cancelled_move_ids
            assert state.positions == [0.0]
    finally:
        release.set()
        robot.set_state_change_callback(None)
        robot.stop(timeout=1.0)


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


def test_cancelled_main_wait_returns_without_spending_its_timeout(robot):
    robot.move_joint_trajectory_async([[1.0]])
    robot.stop()
    robot.initialize()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(robot.wait_move, 10.0, 10.0)
        try:
            assert waiting.result(timeout=0.5) is False
        finally:
            robot.stop(timeout=1.0)


def test_restart_preserves_completed_main_move_id(robot):
    robot.initialize()
    completed = robot.move_joint_trajectory_async([[1.0]], interval=0.001)
    assert robot.wait_move(time_out=1.0)
    robot.stop()
    robot.initialize()
    assert robot.wait_move(time_out=0.1)
    assert robot.query_state().latest_finished_id == completed


def test_simulated_segments_publish_completion_only_at_their_endpoints():
    robot = SimulatedRobotTask([0.0])
    cancelled_move_id = robot.move_joint_trajectory_async([[-1.0]])
    robot.stop()
    observed = []
    finished = threading.Event()

    def observe_state():
        state = robot.query_state()
        if state.jps[0] > 0:
            observed.append((state.jps[0], state.latest_finished_id))
            if state.jps == [4.0]:
                finished.set()

    try:
        robot.initialize()
        robot.set_state_change_callback(observe_state)
        first_move_id = robot.move_joint_trajectory_async(
            [[1.0], [2.0], [3.0], [4.0]], interval=0.001, endpoint_index=[2, 4]
        )
        assert first_move_id > cancelled_move_id
        assert finished.wait(1.0)
        assert observed == [
            (1.0, -1),
            (2.0, first_move_id),
            (3.0, first_move_id),
            (4.0, first_move_id + 1),
        ]
        assert robot.query_state().cancelled_move_ids == (cancelled_move_id,)
    finally:
        robot.set_state_change_callback(None)
        robot.stop(timeout=1.0)


@pytest.mark.parametrize("restart", [False, True])
def test_idle_worker_wakes_for_motion_and_restart(robot, monkeypatch, restart):
    idle = threading.Event()
    original_wait = robot._state_condition.wait

    def observe_idle(timeout=None):
        if timeout is None:
            idle.set()
        return original_wait(timeout)

    monkeypatch.setattr(robot._state_condition, "wait", observe_idle)
    robot.initialize()
    assert idle.wait(1.0)
    if restart:
        robot.stop(timeout=1.0)
        idle.clear()
        robot.initialize()
        assert idle.wait(1.0)
    idle.clear()
    assert robot.move_joint_trajectory_async([[1.0], [2.0]], interval=0.001) == 0
    assert robot.wait_move(time_out=1.0)
    assert robot.query_state().jps[0] == 2.0
    assert idle.wait(1.0)


def test_completed_main_trajectory_is_released_while_robot_is_idle(robot, monkeypatch):
    idle = threading.Event()
    trajectory_reclaimed = threading.Event()
    point_reclaimed = threading.Event()
    trajectory_copies = []
    point_copies = []

    class TrackedPoint(list):
        pass

    class TrackedTrajectory(list):
        def __deepcopy__(self, memo):
            copied = type(self)([type(point)(point) for point in self])
            trajectory_copies.append(
                weakref.ref(copied, lambda _: trajectory_reclaimed.set())
            )
            point_copies.append(
                weakref.ref(copied[-1], lambda _: point_reclaimed.set())
            )
            return copied

    original_wait = robot._state_condition.wait

    def observe_idle(timeout=None):
        if timeout is None:
            idle.set()
        return original_wait(timeout)

    monkeypatch.setattr(robot._state_condition, "wait", observe_idle)
    assert robot.move_joint_trajectory_async(
        TrackedTrajectory([[1.0], TrackedPoint([2.0])]), interval=0.001
    ) == 0
    robot.initialize()
    assert robot.wait_move(time_out=1.0)
    assert idle.wait(1.0)
    assert trajectory_reclaimed.wait(1.0)
    assert point_reclaimed.wait(1.0)
    assert len(trajectory_copies) == 1 and trajectory_copies[0]() is None
    assert len(point_copies) == 1 and point_copies[0]() is None


def test_state_snapshot_is_isolated_from_caller_mutation(robot):
    for value in range(3):
        assert robot.move_joint_trajectory_async([[float(value)]]) == value
    robot.stop()
    snapshot = robot.query_state()
    assert snapshot.cancelled_move_ids == (0, 1, 2)
    snapshot.jps[0] = 99.0
    snapshot.latest_sent_id = 99
    snapshot.cancelled_move_ids += (99,)
    current = robot.query_state()
    assert current.jps[0] == 0.0
    assert current.latest_sent_id == 2
    assert current.cancelled_move_ids == (0, 1, 2)


def test_simulated_snapshot_preserves_custom_state_fields_and_aliases():
    @dataclass
    class State(SimulatedRobotTask.ReducedRobotState):
        details: dict = field(default_factory=dict)

    class Robot(SimulatedRobotTask):
        ReducedRobotState = State

        def __init__(self):
            super().__init__([0.0])
            self._state.details["owner"] = ["simulator"]
            self._state.details["positions"] = self._state.jps

    robot = Robot()
    try:
        assert robot.move_joint_trajectory_async([[1.0]]) == 0
        robot.stop()
        snapshot = robot.query_state()
        assert type(snapshot) is State
        assert snapshot.cancelled_move_ids == (0,)
        assert snapshot.details["owner"] == ["simulator"]
        assert snapshot.details["positions"] is snapshot.jps
        snapshot.details["owner"].append("caller")
        snapshot.jps[0] = 99.0
        current = robot.query_state()
        assert current.details["owner"] == ["simulator"]
        assert current.jps == [0.0]
    finally:
        robot.stop()


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
