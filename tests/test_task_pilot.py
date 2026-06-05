import concurrent.futures
from unittest.mock import MagicMock
from orchestrion.move_sync_option import MoveSyncOption
from orchestrion.task_pilot import TaskPilot


class MockRobotTask:

    def __init__(self):
        self._initialized = False
        self._stopped = False
        self._latest_sent_id = 10
        self._latest_finished_id = 10

    def initialize(self):
        self._initialized = True

    def stop(self):
        self._stopped = True

    def query_state(self):
        return MagicMock(
            latest_sent_id=self._latest_sent_id,
            latest_finished_id=self._latest_finished_id,
        )

    def move_joint_trajectory_async(
        self, motion_target, interval=0.01, endpoint_index=None
    ):
        return 5


class MockGenericTask:

    def __init__(self):
        self._initialized = False
        self._stopped = False
        self._invoked = []

    def initialize(self, executor=None):
        self._initialized = True

    def stop(self):
        self._stopped = True

    def invoke_async(self, request_id, content):
        self._invoked.append((request_id, content))
        return True


def test_initialize():
    robot_task = MockRobotTask()
    task_map = {"task1": MockGenericTask(), "task2": MockGenericTask()}
    supervisor = TaskPilot(robot_task, task_map)
    supervisor.initialize()
    assert robot_task._initialized
    assert task_map["task1"]._initialized
    assert task_map["task2"]._initialized
    print("test_initialize passed")


def test_call_srv_async_no_sync():
    robot_task = MockRobotTask()
    task_map = {"task1": MockGenericTask()}
    supervisor = TaskPilot(robot_task, task_map)
    supervisor.initialize()
    req_id = supervisor.call_srv_async("task1", {"data": 123}, MoveSyncOption.no_sync())
    assert req_id == 0
    print("test_call_srv_async_no_sync passed")


def test_move_joint_trajectory_async():
    robot_task = MockRobotTask()
    task_map = {}
    supervisor = TaskPilot(robot_task, task_map)
    begin_id, end_id = supervisor.move_joint_trajectory_async([[1, 2, 3], [4, 5, 6]])
    assert begin_id == 5 and end_id == 6
    print("test_move_joint_trajectory_async passed")


def test_query_robot_state():
    robot_task = MockRobotTask()
    task_map = {}
    supervisor = TaskPilot(robot_task, task_map)
    state = supervisor.query_robot_state()
    assert state.latest_sent_id == 10
    print("test_query_robot_state passed")


def test_wait_move():
    robot_task = MockRobotTask()
    task_map = {}
    supervisor = TaskPilot(robot_task, task_map)
    supervisor.wait_move(time_out=0.2)
    print("test_wait_move passed")


def test_call_srv_async_sync_latest():
    robot_task = MockRobotTask()
    task_map = {"task1": MockGenericTask()}
    supervisor = TaskPilot(robot_task, task_map)
    supervisor.initialize()
    req_id = supervisor.call_srv_async(
        "task1", {"data": 456}, MoveSyncOption.sync_w_latest_move()
    )
    assert req_id == 0
    print("test_call_srv_async_sync_latest passed")


def test_call_srv_async_sync_explicit_id():
    robot_task = MockRobotTask()
    task_map = {"task1": MockGenericTask()}
    supervisor = TaskPilot(robot_task, task_map)
    supervisor.initialize()
    req_id = supervisor.call_srv_async(
        "task1", {"data": 789}, MoveSyncOption.sync_w_explicit_id(5)
    )
    assert req_id == 0
    print("test_call_srv_async_sync_explicit_id passed")


def test_multiple_requests():
    robot_task = MockRobotTask()
    task_map = {"task1": MockGenericTask()}
    supervisor = TaskPilot(robot_task, task_map)
    supervisor.initialize()
    for i in range(5):
        req_id = supervisor.call_srv_async(
            "task1", {"data": i}, MoveSyncOption.no_sync()
        )
        assert req_id == i
    print("test_multiple_requests passed")


def test_stop():
    robot_task = MockRobotTask()
    task_map = {"task1": MockGenericTask()}
    supervisor = TaskPilot(robot_task, task_map)
    supervisor.initialize()
    supervisor.stop()
    assert robot_task._stopped
    assert task_map["task1"]._stopped
    print("test_stop passed")


def test_executor_shutdown():
    robot_task = MockRobotTask()
    task_map = {"task1": MockGenericTask()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        supervisor = TaskPilot(robot_task, task_map, executor_owned=executor)
        supervisor.initialize()
        supervisor.stop()
    print("test_executor_shutdown passed")


if __name__ == "__main__":
    test_initialize()
    test_call_srv_async_no_sync()
    test_move_joint_trajectory_async()
    test_query_robot_state()
    test_wait_move()
    test_call_srv_async_sync_latest()
    test_call_srv_async_sync_explicit_id()
    test_multiple_requests()
    test_stop()
    test_executor_shutdown()
    print("All TaskPilot tests passed.")
