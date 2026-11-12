import concurrent.futures
import gc
import queue
import threading
import time
import weakref
from fractions import Fraction

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


@pytest.mark.parametrize(
    "interval", [0.0, -0.01, float("nan"), float("inf"), "0.01", True]
)
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


@pytest.mark.parametrize(
    "interval", [0.0, -0.01, float("nan"), float("inf"), "0.01", True]
)
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
        ("0.1", 0.01),
        (True, 0.01),
        (0.1, 0.0),
        (0.1, -0.01),
        (0.1, float("nan")),
        (0.1, "0.01"),
        (0.1, True),
    ],
)
def test_wait_move_rejects_invalid_timing(time_out, interval):
    with pytest.raises(ValueError):
        make_robot().wait_move(time_out=time_out, interval=interval)


@pytest.mark.parametrize(
    "timeout", [-0.1, float("nan"), float("inf"), "0.1", True]
)
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


@pytest.mark.parametrize("submodule", [False, True])
def test_idle_robot_wakes_on_new_motion_without_waiting_for_scheduler_tick(
    monkeypatch, submodule
):
    idle = threading.Event()
    updated = threading.Event()
    robot = make_robot(interval=10.0)
    original_wait = robot._state_condition.wait

    def observe_wait(timeout=None):
        if timeout is None:
            idle.set()
        return original_wait(timeout)

    monkeypatch.setattr(robot._state_condition, "wait", observe_wait)
    monkeypatch.setattr(
        robot, "_on_robot_state_bg_thread", lambda *_: updated.set()
    )
    robot.initialize()
    try:
        assert idle.wait(1.0)
        if submodule:
            assert robot.move_submodule_trajectory_async("gripper", [[0.8]]) == 0
        else:
            assert robot.move_joint_trajectory_async([[0.1, 0.2, 0.3]]) == 0
        assert updated.wait(1.0)
    finally:
        robot.stop(timeout=1.0)


def test_motion_submitted_after_the_last_queue_poll_is_not_missed(monkeypatch):
    polled_empty = threading.Event()
    submitted = threading.Event()
    updated = threading.Event()

    class PausedEmptyQueue(queue.SimpleQueue):
        def get_nowait(self):
            try:
                return super().get_nowait()
            except queue.Empty:
                if not polled_empty.is_set():
                    polled_empty.set()
                    assert submitted.wait(1.0)
                raise

    robot = make_robot(interval=10.0)
    robot._main_task_queue = PausedEmptyQueue()
    monkeypatch.setattr(robot, "_on_robot_state_bg_thread", lambda *_: updated.set())
    robot.initialize()
    try:
        assert polled_empty.wait(1.0)
        assert robot.move_joint_trajectory_async([[0.1, 0.2, 0.3]]) == 0
        submitted.set()
        assert updated.wait(1.0)
        assert robot.wait_move(time_out=1.0)
    finally:
        submitted.set()
        robot.stop(timeout=1.0)


def test_idle_worker_releases_completed_submodule_trajectory(monkeypatch):
    idle = threading.Event()
    copies = []

    class TrackedTrajectory(list):
        def __deepcopy__(self, memo):
            copied = type(self)([point.copy() for point in self])
            copies.append(weakref.ref(copied))
            return copied

    robot = make_robot()
    original_wait = robot._state_condition.wait

    def observe_idle(timeout=None):
        if timeout is None:
            idle.set()
        return original_wait(timeout)

    monkeypatch.setattr(robot._state_condition, "wait", observe_idle)
    move_id = robot.move_submodule_trajectory_async(
        "gripper", TrackedTrajectory([[0.8]])
    )
    robot.initialize()
    try:
        assert robot.wait_submodule_move("gripper", move_id, timeout=1.0)
        assert idle.wait(1.0)
        gc.collect()
        assert len(copies) == 1 and copies[0]() is None
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


def test_batch_submodule_submission_is_atomic_and_completes_in_parallel():
    robot = RobotTaskStub(
        [0.0] * 5,
        SubModuleTask("torso", 0, 1),
        [SubModuleTask("left_arm", 1, 2), SubModuleTask("right_arm", 3, 2)],
        interval=0.002,
    )
    rejected = robot.move_submodules_trajectories_async(
        {"left_arm": [[0.1, 0.2]], "right_arm": [[0.1]]}
    )
    assert rejected == {}
    assert robot.query_submodule_state("left_arm").latest_sent_id == -1
    assert robot.query_submodule_state("right_arm").latest_sent_id == -1

    robot.initialize()
    try:
        moves = robot.move_submodules_trajectories_async(
            {
                "left_arm": [[0.1, 0.2], [0.2, 0.3]],
                "right_arm": [[-0.1, -0.2], [-0.2, -0.3]],
            },
            interval=0.002,
        )
        assert moves == {"left_arm": 0, "right_arm": 0}
        assert robot.wait_submodule_moves(moves, timeout=0.5)
    finally:
        robot.stop()


def test_batch_submodule_copy_failure_does_not_allocate_any_move_ids():
    class FailingCopyList(list):
        def __deepcopy__(self, memo):
            raise RuntimeError("copy failed")

    robot = RobotTaskStub(
        [0.0] * 3,
        SubModuleTask("torso", 0, 1),
        [SubModuleTask("left_arm", 1, 1), SubModuleTask("right_arm", 2, 1)],
    )
    assert (
        robot.move_submodules_trajectories_async(
            {"left_arm": [[0.1]], "right_arm": FailingCopyList([[0.1]])}
        )
        == {}
    )
    assert robot.query_submodule_state("left_arm").latest_sent_id == -1
    assert robot.query_submodule_state("right_arm").latest_sent_id == -1


def test_cancelled_history_is_independent_per_arm_and_survives_restart():
    robot = RobotTaskStub(
        [0.0] * 3,
        SubModuleTask("torso", 0, 1),
        [SubModuleTask("left", 1, 1), SubModuleTask("right", 2, 1)],
        interval=0.001,
    )
    try:
        first = robot.move_submodules_trajectories_async(
            {"left": [[0.1]], "right": [[0.2]]}, interval=0.001
        )
        second = robot.move_submodules_trajectories_async(
            {"left": [[0.3]], "right": [[0.4]]}, interval=0.001
        )
        assert robot.cancel_submodule_move("left", first["left"])
        assert robot.cancel_submodule_moves({"right": second["right"]}) == {
            "right": True
        }
        robot.initialize()
        assert robot.wait_submodule_move("left", second["left"], timeout=1.0)
        assert robot.wait_submodule_move("right", first["right"], timeout=1.0)
        robot.stop()
        robot.initialize()
        resumed = robot.move_submodules_trajectories_async(
            {"left": [[0.5]], "right": [[0.6]]}, interval=0.001
        )
        assert robot.wait_submodule_moves(resumed, timeout=1.0)
        states = robot.query_submodule_states()
        assert states["left"].cancelled_move_ids == (first["left"],)
        assert states["right"].cancelled_move_ids == (second["right"],)
        assert not robot.wait_submodule_move("left", first["left"], timeout=0)
        assert robot.wait_submodule_move("right", first["right"], timeout=0)
    finally:
        robot.stop()


def test_cancellation_snapshots_follow_single_batch_stop_and_restart():
    robot = RobotTaskStub(
        [0.0] * 3,
        SubModuleTask("torso", 0, 1),
        [SubModuleTask("left", 1, 1), SubModuleTask("right", 2, 1)],
        interval=0.001,
    )
    try:
        for _ in range(2):
            robot.move_joint_trajectory_async([[0.1]], interval=0.001)
            robot.move_submodules_trajectories_async(
                {"left": [[0.2]], "right": [[0.3]]}, interval=0.001
            )
        initial_main = robot.query_state()
        initial_submodules = robot.query_submodule_states()
        assert robot.cancel_submodule_move("left", 0)
        after_single = robot.query_submodule_states()
        assert after_single["left"].cancelled_move_ids == (0,)
        assert robot.cancel_submodule_moves({"left": 1, "right": 0}) == {
            "left": True,
            "right": True,
        }
        after_batch = robot.query_submodule_states()
        assert after_batch["left"].cancelled_move_ids == (0, 1)
        assert after_batch["right"].cancelled_move_ids == (0,)
        assert after_single["left"].cancelled_move_ids == (0,)
        assert initial_main.cancelled_move_ids == ()
        assert all(not state.cancelled_move_ids for state in initial_submodules.values())

        robot.stop()
        assert robot.query_state().cancelled_move_ids == (0, 1)
        assert robot.query_submodule_state("right").cancelled_move_ids == (0, 1)
        assert after_batch["right"].cancelled_move_ids == (0,)
        robot.initialize()
        assert robot.move_joint_trajectory_async([[0.4]], interval=0.001) == 2
        resumed = robot.move_submodules_trajectories_async(
            {"left": [[0.5]], "right": [[0.6]]}, interval=0.001
        )
        assert robot.wait_move(time_out=1.0)
        assert robot.wait_submodule_moves(resumed, timeout=1.0)
        assert robot.query_state().cancelled_move_ids == (0, 1)
        assert all(
            state.cancelled_move_ids == (0, 1)
            for state in robot.query_submodule_states().values()
        )
    finally:
        robot.stop()


def test_fraction_scheduler_and_submodule_intervals():
    interval = Fraction(1, 1000)
    robot = RobotTaskStub(
        [0.0] * 3,
        SubModuleTask("torso", 0, 1),
        [SubModuleTask("left", 1, 1), SubModuleTask("right", 2, 1)],
        interval=interval,
    )
    robot.initialize()
    try:
        first = robot.move_submodule_trajectory_async("left", [[0.1]], interval)
        assert robot.wait_submodule_move("left", first, timeout=Fraction(1))
        moves = robot.move_submodules_trajectories_async(
            {"left": [[0.2]], "right": [[0.3]]}, interval
        )
        assert robot.wait_submodule_moves(moves, timeout=Fraction(1))
        states = robot.query_submodule_states()
        assert states["left"].positions == [0.2]
        assert states["right"].positions == [0.3]
        robot.stop(timeout=Fraction(1))
    finally:
        robot.stop()


def test_concurrent_batch_submissions_keep_dual_arm_move_ids_paired():
    robot = RobotTaskStub(
        [0.0] * 3,
        SubModuleTask("torso", 0, 1),
        [SubModuleTask("left_arm", 1, 1), SubModuleTask("right_arm", 2, 1)],
    )

    def submit(value):
        return robot.move_submodules_trajectories_async(
            {"left_arm": [[value]], "right_arm": [[-value]]}
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        batches = list(executor.map(submit, [index / 100.0 for index in range(50)]))
    assert all(batch["left_arm"] == batch["right_arm"] for batch in batches)
    assert sorted(batch["left_arm"] for batch in batches) == list(range(50))
    states = robot.query_submodule_states(["left_arm", "right_arm"])
    assert states["left_arm"].latest_sent_id == 49
    assert states["right_arm"].latest_sent_id == 49
    robot.stop()


def test_batch_is_not_visible_to_scheduler_until_every_arm_is_enqueued():
    class BlockingQueue:
        def __init__(self):
            self.queue = queue.SimpleQueue()
            self.first_put = threading.Event()
            self.release_put = threading.Event()
            self.get_called = threading.Event()
            self.put_count = 0

        def put(self, item):
            self.put_count += 1
            self.queue.put(item)
            if self.put_count == 1:
                self.get_called.clear()
                self.first_put.set()
                assert self.release_put.wait(0.5)

        def get_nowait(self):
            self.get_called.set()
            return self.queue.get_nowait()

        def empty(self):
            return self.queue.empty()

    robot = RobotTaskStub(
        [0.0] * 3,
        SubModuleTask("torso", 0, 1),
        [SubModuleTask("left_arm", 1, 1), SubModuleTask("right_arm", 2, 1)],
        interval=0.002,
    )
    blocking_queue = BlockingQueue()
    robot._sub_task_queue = blocking_queue
    robot.initialize()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            submitted = executor.submit(
                robot.move_submodules_trajectories_async,
                {"left_arm": [[0.1]], "right_arm": [[-0.1]]},
                0.002,
            )
            assert blocking_queue.first_put.wait(0.5)
            time.sleep(0.02)
            assert not blocking_queue.get_called.is_set()
            blocking_queue.release_put.set()
            moves = submitted.result(timeout=0.5)
        assert robot.wait_submodule_moves(moves, timeout=0.5)
    finally:
        blocking_queue.release_put.set()
        robot.stop()


def test_query_submodule_states_returns_selected_atomic_copies():
    robot = make_robot()
    states = robot.query_submodule_states(["gripper"])
    assert tuple(states) == ("gripper",)
    states["gripper"].positions[0] = 99.0
    assert robot.query_submodule_state("gripper").positions == [0.0]
    with pytest.raises(KeyError, match="Unknown submodule"):
        robot.query_submodule_states(["missing"])


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


def test_concurrent_initialize_and_stop_are_serialized():
    robot = make_robot()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(robot.initialize) for _ in range(16)]
        for future in futures:
            future.result(timeout=0.5)
        thread = robot._bg_thread
        assert thread is not None and thread.is_alive()

        futures = [executor.submit(robot.stop) for _ in range(16)]
        for future in futures:
            future.result(timeout=0.5)
    assert robot._bg_thread is thread
    assert not thread.is_alive()
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


@pytest.mark.parametrize(
    "timeout", [-0.1, float("nan"), float("inf"), "0.1", True]
)
def test_wait_submodule_move_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        make_robot().wait_submodule_move("gripper", 0, timeout=timeout)


@pytest.mark.parametrize("move_id", [True, 1.5, "0"])
def test_submodule_move_apis_reject_non_integer_move_ids(move_id):
    robot = make_robot()
    with pytest.raises(TypeError, match="move_id"):
        robot.wait_submodule_move("gripper", move_id)
    with pytest.raises(TypeError, match="move_id"):
        robot.cancel_submodule_move("gripper", move_id)


def test_submodule_move_apis_reject_negative_and_unknown_move_ids():
    robot = make_robot()
    with pytest.raises(ValueError, match="non-negative"):
        robot.wait_submodule_move("gripper", -1)
    with pytest.raises(ValueError, match="non-negative"):
        robot.cancel_submodule_move("gripper", -1)
    with pytest.raises(KeyError, match="Unknown move"):
        robot.wait_submodule_move("gripper", 0)
    assert not robot.cancel_submodule_move("gripper", 0)


def test_submodule_move_can_be_cancelled():
    robot = make_robot(interval=0.01)
    robot.initialize()
    try:
        move_id = robot.move_submodule_trajectory_async(
            "gripper", [[value / 100.0] for value in range(50)]
        )
        assert robot.cancel_submodule_move("gripper", move_id)
        assert not robot.cancel_submodule_move("gripper", move_id)
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


def test_wait_submodule_moves_waits_in_parallel_with_shared_timeout():
    robot = RobotTaskStub(
        [0.0] * 5,
        SubModuleTask("torso", 0, 1),
        [SubModuleTask("left_arm", 1, 2), SubModuleTask("right_arm", 3, 2)],
        interval=0.002,
    )
    robot.initialize()
    try:
        moves = {
            "left_arm": robot.move_submodule_trajectory_async(
                "left_arm", [[0.1, 0.2], [0.2, 0.3]], interval=0.002
            ),
            "right_arm": robot.move_submodule_trajectory_async(
                "right_arm", [[-0.1, -0.2], [-0.2, -0.3]], interval=0.002
            ),
        }
        assert robot.wait_submodule_moves(moves, timeout=0.5)
        assert robot.query_submodule_state("left_arm").positions == [0.2, 0.3]
        assert robot.query_submodule_state("right_arm").positions == [-0.2, -0.3]
    finally:
        robot.stop()


def test_wait_submodule_moves_fails_when_any_move_is_cancelled():
    robot = RobotTaskStub(
        [0.0] * 3,
        SubModuleTask("torso", 0, 1),
        [SubModuleTask("left_arm", 1, 1), SubModuleTask("right_arm", 2, 1)],
    )
    left = robot.move_submodule_trajectory_async("left_arm", [[0.1]])
    right = robot.move_submodule_trajectory_async("right_arm", [[0.1]])
    assert robot.cancel_submodule_move("right_arm", right)
    assert not robot.wait_submodule_moves(
        {"left_arm": left, "right_arm": right}, timeout=0
    )


def test_cancel_submodule_moves_cancels_group_atomically():
    robot = RobotTaskStub(
        [0.0] * 3,
        SubModuleTask("torso", 0, 1),
        [SubModuleTask("left_arm", 1, 1), SubModuleTask("right_arm", 2, 1)],
    )
    moves = robot.move_submodules_trajectories_async(
        {"left_arm": [[0.1]], "right_arm": [[0.1]]}
    )
    assert robot.cancel_submodule_moves(moves) == {
        "left_arm": True,
        "right_arm": True,
    }
    assert robot.cancel_submodule_moves(moves) == {
        "left_arm": False,
        "right_arm": False,
    }
    assert not robot.wait_submodule_moves(moves, timeout=0)


def test_cancel_submodule_moves_validates_all_names_before_mutating():
    robot = make_robot()
    move_id = robot.move_submodule_trajectory_async("gripper", [[0.1]])
    with pytest.raises(KeyError, match="missing"):
        robot.cancel_submodule_moves({"gripper": move_id, "missing": 0})
    assert robot.cancel_submodule_move("gripper", move_id)


@pytest.mark.parametrize("moves", [[], {"left_arm": True}, {1: 0}])
def test_wait_submodule_moves_validates_mapping(moves):
    robot = make_robot()
    expected = TypeError
    with pytest.raises(expected):
        robot.wait_submodule_moves(moves)


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True, "1"])
def test_wait_submodule_moves_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        make_robot().wait_submodule_moves({}, timeout=timeout)


def test_wait_submodule_moves_rejects_unknown_or_unsent_moves():
    robot = make_robot()
    with pytest.raises(KeyError, match="Unknown submodule"):
        robot.wait_submodule_moves({"missing": 0})
    with pytest.raises(KeyError, match="Unknown move"):
        robot.wait_submodule_moves({"gripper": 0})


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


def test_submodule_completion_and_cancellation_notify_state_callback():
    robot = make_robot(interval=0.002)
    completed = threading.Event()
    cancelled = threading.Event()

    def observe():
        state = robot.query_submodule_state("gripper")
        if state.latest_finished_id >= 0:
            completed.set()
        if state.cancelled_move_ids:
            cancelled.set()

    robot.set_state_change_callback(observe)
    robot.initialize()
    try:
        move_id = robot.move_submodule_trajectory_async("gripper", [[0.2]])
        assert robot.wait_submodule_move("gripper", move_id, timeout=0.5)
        assert completed.wait(0.2)
        next_id = robot.move_submodule_trajectory_async(
            "gripper", [[value / 100.0] for value in range(50)]
        )
        assert robot.cancel_submodule_move("gripper", next_id)
        assert cancelled.wait(0.2)
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


@pytest.mark.parametrize("failure_at", ["initial", "main", "submodule"])
def test_worker_callback_failure_cancels_motion_and_allows_explicit_restart(
    monkeypatch, failure_at
):
    robot = make_robot(interval=0.001)
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    callback_name = (
        "_on_initial_state_bg_thread"
        if failure_at == "initial"
        else "_on_robot_state_bg_thread"
    )
    original_callback = getattr(robot, callback_name)
    original_wait = robot._state_condition.wait

    def fail_callback(*args):
        entered.set()
        assert release.wait(1.0)
        raise RuntimeError("device callback failed")

    def observe_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(robot, callback_name, fail_callback)
    monkeypatch.setattr(robot._state_condition, "wait", observe_wait)
    is_submodule = failure_at == "submodule"
    if is_submodule:
        move_id = robot.move_submodule_trajectory_async("gripper", [[0.5]])
    else:
        move_id = robot.move_joint_trajectory_async([[0.1, 0.2, 0.3]])
    robot.initialize()
    try:
        assert entered.wait(1.0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            if is_submodule:
                result = executor.submit(robot.wait_submodule_move, "gripper", move_id)
            else:
                result = executor.submit(robot.wait_move)
            try:
                assert waiting.wait(1.0)
                release.set()
                assert result.result(timeout=1.0) is False
            finally:
                release.set()
                robot.stop(timeout=1.0)

        state = (
            robot.query_submodule_state("gripper")
            if is_submodule
            else robot.query_state()
        )
        assert move_id in state.cancelled_move_ids
        assert robot.move_joint_trajectory_async([[0.0, 0.0, 0.0]]) == -1
        assert robot.move_submodule_trajectory_async("gripper", [[0.0]]) == -1

        monkeypatch.setattr(robot, callback_name, original_callback)
        robot.initialize()
        if is_submodule:
            later_id = robot.move_submodule_trajectory_async("gripper", [[0.8]])
            assert robot.wait_submodule_move("gripper", later_id, timeout=1.0)
            state = robot.query_submodule_state("gripper")
        else:
            later_id = robot.move_joint_trajectory_async([[0.4, 0.5, 0.6]])
            assert robot.wait_move(time_out=1.0)
            state = robot.query_state()
        assert later_id > move_id
        assert move_id in state.cancelled_move_ids
        assert later_id not in state.cancelled_move_ids
    finally:
        release.set()
        robot.stop(timeout=1.0)
