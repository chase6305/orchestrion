import time
import queue
import threading
import concurrent.futures
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from orchestrion.tasks.generic_task import GenericTask
from orchestrion.tasks.reduced_robot_task_interface import ReducedRobotTaskInterface
from orchestrion.tasks.modular_reduced_robot_task import ModularReducedRobotTask
from orchestrion.move_sync_option import MoveSyncOption
from orchestrion.utils.logger import logger


class TaskPilot(object):
    """
    Supervisor class for managing robot task and peripheral device tasks.
    Handles background task scheduling, synchronization, and coordination.
    """

    @dataclass
    class BackgroundRequest(object):
        """
        Data class for internal background task requests.

        Attributes:
            srv_name (str): task name.
            associated_move_id (int): Associated move ID for synchronization.
            request_id (int): Unique request ID.
            content (Optional[Dict]): Optional request content.
        """

        srv_name: str
        associated_move_id: int = -1
        request_id: int = -1
        content: Optional[Dict] = None

    def __init__(
        self,
        robot_task: ModularReducedRobotTask,
        task_map: Optional[Dict[str, GenericTask]] = None,
        executor_owned: Optional[concurrent.futures.ThreadPoolExecutor] = None,
    ):
        """
        Initialize the TaskPilot.

        Args:
            robot_task (ModularReducedRobotTask): The main robot task.
            task_map (Dict[str, GenericTask]): Peripheral device task map.
            executor_owned (Optional[ThreadPoolExecutor]): Shared thread pool executor.
        """
        self._robot_task = robot_task  # Robot main task
        self._task_map = (
            task_map if task_map is not None else {}
        )  # Peripheral device task map
        self._task_queue = queue.SimpleQueue()  # Task queue for background requests
        self._next_request_id: int = 0  # Next request ID
        self._bg_thread: Optional[threading.Thread] = None  # Background thread
        self._executor = executor_owned  # Shared thread pool
        self._stop = False  # Stop flag

    @property
    def task_map(self):
        """
        Get the peripheral device task map.
        Returns:
            Dict[str, GenericTask]: The task map.
        """
        return self._task_map

    @property
    def robot_task(self):
        """
        Get the main robot task.
        Returns:
            ModularReducedRobotTask: The robot task instance.
        """
        return self._robot_task

    def initialize(self):
        """
        Initialize the supervisor, robot task, and all peripheral tasks.
        Starts the background thread.
        """
        if self._bg_thread is not None and self._bg_thread.is_alive():
            logger.warning("Supervisor is already initialized.")
            return

        # Init robot task
        self._robot_task.initialize()

        # Init other tasks
        for _, v in self._task_map.items():
            v.initialize(executor=self._executor)

        # Start background thread for supervision
        self._bg_thread = threading.Thread(target=self._bg_thread_loop)
        self._bg_thread.start()

    def stop(self):
        """
        Stop the supervisor, background thread, all tasks, and the robot task.
        """
        if self._bg_thread is None or not self._bg_thread.is_alive():
            logger.warning("Supervisor is not running or already stopped.")
            return

        self._stop = True
        self._bg_thread.join()

        # Stop all the tasks
        for _, v in self._task_map.items():
            v.stop()
        self._executor.shutdown(cancel_futures=True)

        # Stop the robot task
        self._robot_task.stop()

    def call_srv_async(
        self,
        srv_name: str,
        content: Optional[Dict] = None,
        sync_option: MoveSyncOption = MoveSyncOption.sync_w_latest_move(),
    ) -> int:
        """
        Asynchronously call a peripheral service/task, optionally synchronized with a robot move.

        Args:
            srv_name (str): Name of the service/task.
            content (Optional[Dict]): Optional request content.
            sync_option (MoveSyncOption): Synchronization option.

        Returns:
            int: The request ID assigned to this call.
        """
        # Make request id
        request_id = self._next_request_id
        self._next_request_id += 1

        # Make move id
        associated_move_id: int = -1
        if sync_option.need_sync:
            if sync_option.associated_move_id >= 0:
                associated_move_id = sync_option.associated_move_id
            else:
                # Query from latest move id
                local_robot_state: Optional[
                    ReducedRobotTaskInterface.ReducedRobotState
                ] = None
                while local_robot_state is None:
                    local_robot_state = self._robot_task.query_state()
                    # Do NOT need to sleep here, as this is likely due to lock contention

                # Get the id
                assert local_robot_state is not None
                associated_move_id = local_robot_state.latest_sent_id

        # Make task
        task = TaskPilot.BackgroundRequest(
            srv_name=srv_name,
            associated_move_id=associated_move_id,
            request_id=request_id,
            content=content,
        )
        self._task_queue.put(task)

        # Done
        return request_id

    def move_joint_trajectory_async(
        self,
        motion_target: List[List[float]],
        interval: float = 0.01,
        endpoint_index: Optional[List[int]] = None,
    ) -> Tuple[int, int]:
        """
        Execute a robot joint trajectory movement asynchronously.

        Args:
            motion_target (List[List[float]]): List of joint positions for the trajectory.
            interval (float): Time interval between points.
            endpoint_index (Optional[List[int]]): Segment endpoints.

        Returns:
            Tuple[int, int]: The range of movement IDs [begin_id, end_id).
                             end_id is NOT included.
        """
        n_segments = 1 if endpoint_index is None else len(endpoint_index)
        move_id_begin = self._robot_task.move_joint_trajectory_async(
            motion_target=motion_target,
            interval=interval,
            endpoint_index=endpoint_index,
        )
        if move_id_begin < 0:
            return move_id_begin, move_id_begin

        # Correct
        return move_id_begin, move_id_begin + n_segments

    def query_robot_state(self):
        """
        Query the current robot state.

        Returns:
            ReducedRobotTaskInterface.ReducedRobotState: The current robot state.
        """
        return self._robot_task.query_state()

    def wait_move(self, time_out: float = -1, interval: float = 0.05) -> bool:
        """
        Wait for all robot movements to finish or until timeout.

        Args:
            time_out (float): Maximum time to wait in seconds.
                              If negative, wait indefinitely. Default is -1.
            interval (float): Polling interval in seconds. Default is 0.05s.

        Returns:
            bool: True if movements finished before timeout, False otherwise.
        """
        return self._robot_task.wait_move(time_out=time_out, interval=interval)

    def _bg_thread_loop(self, sleep_interval: float = 0.05, idle_interval: float = 0.1):
        """
        Background thread loop for task scheduling and synchronization.
        Handles both synchronized and non-synchronized tasks, and executes them when ready.

        Args:
            sleep_interval (float): Sleep interval between checks when active.
            idle_interval (float): Sleep interval when idle.
        """
        local_tasks_no_sync = deque()
        local_tasks_sync = deque()
        while not self._stop:
            # Pop task to local stack
            while not self._stop:
                try:
                    elem: TaskPilot.BackgroundRequest = self._task_queue.get_nowait()
                    if elem.associated_move_id >= 0:
                        local_tasks_sync.append(elem)
                    else:
                        local_tasks_no_sync.append(elem)
                except queue.Empty:
                    break

            if self._stop:
                break

            # Run all no sync tasks immediately
            while local_tasks_no_sync:
                task: TaskPilot.BackgroundRequest = local_tasks_no_sync.popleft()
                assert task.associated_move_id < 0
                if task.srv_name in self._task_map:
                    task: GenericTask = self._task_map[task.srv_name]
                    task.invoke_async(task.request_id, task.content)

            if not local_tasks_sync:
                # No sync tasks, sleep for a while
                time.sleep(idle_interval)
                continue

            # Query robot status and check for synced tasks
            robot_state = self._robot_task.query_state()
            if robot_state is None:
                # To next loop
                time.sleep(idle_interval)
                continue

            # Execute the tasks that depend on robot state
            remaining_tasks = deque()
            for task in local_tasks_sync:
                assert task.associated_move_id >= 0
                if task.associated_move_id > robot_state.latest_finished_id:
                    remaining_tasks.append(task)
                    continue

                # Else execute this
                if task.srv_name in self._task_map:
                    request_id, content = task.request_id, task.content
                    mapped_task: GenericTask = self._task_map[task.srv_name]
                    # TODO: Handle exceptions here
                    mapped_task.invoke_async(request_id, content)

            # Update the tasks
            local_tasks_sync = remaining_tasks
            # To next loop
            time.sleep(sleep_interval)
