import concurrent.futures
import time

import pytest

from orchestrion.move_sync_option import MoveSyncOption
from orchestrion.task_pilot import TaskPilot
from orchestrion.tasks.reduced_robot_task_interface import ReducedRobotTaskInterface


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
