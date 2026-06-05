import time
import copy
import queue
import threading
from typing import List, Optional, Dict
from abc import abstractmethod

from dataclasses import dataclass
from orchestrion.tasks.reduced_robot_task_interface import ReducedRobotTaskInterface
from orchestrion.utils.logger import logger


class SubModuleTask(object):

    def __init__(self, name: str = "", dof_begin: int = 0, n_dof: int = 0):
        """
        Initialize a SubModuleTask.

        Args:
            name (str): Name of the sub-module.
            dof_begin (int): Starting index of the DOF for this sub-module.
            n_dof (int): Number of DOF for this sub-module.
        """
        self.name: str = name
        self.dof_begin: int = dof_begin
        self.n_dof: int = n_dof


@dataclass
class SubModuleMovementRequest(object):
    """
    Data class representing a movement request for the robot.
    """

    submodule_name: str
    motion_target: List[List[float]]
    interval: float = 0.01


class ModularReducedRobotTask(ReducedRobotTaskInterface):
    """
    ModularReducedRobotTask implements a reduced robot task with modular sub-components.
    It extends the ReducedRobotTaskInterface to support multiple sub-modules.
    """

    def __init__(
        self,
        init_full_q: List[float],
        main_task: SubModuleTask,
        submodule_tasks: List[SubModuleTask],
        interval: float = 0.01,
    ):
        super().__init__()
        self._lock = threading.Lock()
        self._current_full_q = init_full_q.copy()
        self._finished_move_id = -1
        self._latest_sent_id = -1
        self._interval = interval

        # Queues and threads
        self._main_task_queue = queue.SimpleQueue()
        self._main_task = main_task
        self._bg_thread = None
        self._stop: bool = False

        # Sub-module management
        self._sub_task_queue = queue.SimpleQueue()
        self._sub_task_map: Dict[str, SubModuleTask] = dict()
        for elem in submodule_tasks:
            self._sub_task_map[elem.name] = elem

    def initialize(self):
        """
        Initialize and start the background thread for processing movement requests.
        """
        if self._bg_thread is not None and self._bg_thread.is_alive():
            logger.warning("Background thread is already running.")
            return

        self._bg_thread = threading.Thread(target=self._bg_thread_loop)
        self._bg_thread.start()

    def stop(self):
        """
        Stop the background thread and wait for it to finish.
        """
        self._stop = True
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join()

    def move_joint_trajectory_async(
        self,
        motion_target: List[List[float]],
        interval: float = 0.01,
        endpoint_index: Optional[List[int]] = None,
    ) -> int:
        """
        Move the robot along a joint trajectory asynchronously.

        Args:
            motion_target (List[List[float]]): Target joint positions for the trajectory.
            interval (float): Time interval between points.
            endpoint_index (Optional[List[int]]): Segment endpoints.

        Returns:
            int: Move ID if the request was accepted, -1 otherwise.
        """
        # if abs(interval - self._interval) > 1e-4:
        #     logger.warning(
        #         f"The specified interval {interval} "
        #         f"differs from the initialized interval {self._interval}. "
        #         "Using the initialized interval.")
        #     return False

        move_id_begin = self._latest_sent_id + 1
        assigned_endpoint_idx = (
            [len(motion_target)] if endpoint_index is None else endpoint_index
        )
        if not self._check_assigned_endpoint_index(
            motion_target, assigned_endpoint_idx
        ):
            return -1

        # Push to queue
        request = self.MovementRequest(
            motion_target=motion_target,
            interval=interval,
            motion_id_begin=move_id_begin,
            endpoint_index=assigned_endpoint_idx,
        )
        self._main_task_queue.put(request)
        self._latest_sent_id += len(assigned_endpoint_idx)
        return move_id_begin

    def move_submodule_async(
        self,
        submodule_name: str,
        motion_target: List[List[float]],
        interval: float = 0.01,
    ) -> int:
        """
        Move a sub-module asynchronously.

        Args:
            submodule_name (str): Name of the sub-module to move.
            motion_target (List[List[float]]): Target joint positions for the sub-module.
            interval (float): Time interval between points.

        Returns:
            int: Move ID if the request was accepted, -1 otherwise.
        """
        # if abs(interval - self._interval) > 1e-4:
        #     logger.warning(
        #         f"The specified interval {interval} "
        #         f"differs from the initialized interval {self._interval}. "
        #         "Using the initialized interval.")
        #     return False

        # Push to queue
        request = SubModuleMovementRequest(
            submodule_name=submodule_name,
            motion_target=motion_target,
            interval=self._interval,
        )
        self._sub_task_queue.put(request)
        return True

    def query_state(self) -> Optional[ReducedRobotTaskInterface.ReducedRobotState]:
        """
        Query the current state of the robot.

        Returns:
            Optional[ReducedRobotTaskInterface.ReducedRobotState]: The current robot state.
        """
        state: Optional[ReducedRobotTaskInterface.ReducedRobotState] = None
        with self._lock:
            state = self.ReducedRobotState(
                jps=self._current_full_q,
                latest_finished_id=self._finished_move_id,
                latest_sent_id=self._latest_sent_id,
            )
        return state

    def wait_move(self, time_out: float = -1, interval: float = 0.05) -> bool:
        """
        Wait for all robot movements to finish or until timeout.

        Args:
            time_out (float): Maximum time to wait in seconds. If negative, wait indefinitely. Default is -1.
            interval (float): Polling interval in seconds. Default is 0.05s.

        Returns:
            bool: True if movements finished before timeout, False otherwise.
        """
        sent_id = self.query_state().latest_sent_id
        count = 0
        while True:
            state = self.query_state()
            if state.latest_finished_id >= sent_id:
                return True

            count += 1
            time.sleep(interval)
            if 0 < time_out < count * interval:
                return False

    @abstractmethod
    def _on_robot_state_bg_thread(self, jps: List[float], jps_move_id: int):
        """
        Callback for updating the main robot state in the background thread.

        Args:
            jps (List[float]): Current joint positions.
            jps_move_id (int): Move ID associated with the joint positions.
        """
        pass

    @abstractmethod
    def _on_initial_state_bg_thread(self, jps: List[float]):
        """
        Callback for initial state update of the main robot in the background thread.

        Args:
            jps (List[float]): Current joint positions.
        """
        pass

    def _bg_thread_loop(self):
        """
        Main loop of the background thread.
        Responsible for executing movement requests and updating robot state.
        """
        # Local copy of joint positions and finished move id
        current_full_jps: List[float] = copy.deepcopy(self._current_full_q)
        finished_move_id: int = -1
        self._on_initial_state_bg_thread(current_full_jps)

        # State for the current movement
        current_move: Optional[self.MovementRequest] = None
        idx_in_current_move: int = -1
        next_endpoint2check: int = 0
        next_finished_move_id: int = -1

        @dataclass
        class SubTaskState(object):
            idx_in_current_move: int = -1

        # Init for local queue
        local_submodule_task_queue: Dict[str, List[SubModuleMovementRequest]] = dict()
        local_submodule_task_state: Dict[str, SubTaskState] = dict()
        for task_name in self._sub_task_map.keys():
            local_submodule_task_queue[task_name] = list()
            local_submodule_task_state[task_name] = SubTaskState(idx_in_current_move=-1)
        submodule_current_move = None

        # Execution loop
        while not self._stop:
            # 1. If no current move, try to get a new move from the queue
            if current_move is None:
                try:
                    elem: self.MovementRequest = self._main_task_queue.get_nowait()
                    current_move = elem
                    idx_in_current_move = 0
                    next_endpoint2check = 0
                    next_finished_move_id = elem.motion_id_begin
                    assert self._check_assigned_endpoint_index(
                        elem.motion_target, elem.endpoint_index
                    )
                except queue.Empty:
                    pass

            # 2. get tasks from _task_queue_submodules
            pop_submodule_tasks: List[SubModuleMovementRequest] = list()
            while True:
                subtask_elem: Optional[SubModuleMovementRequest] = None
                try:
                    subtask_elem = self._sub_task_queue.get_nowait()
                except queue.Empty:
                    subtask_elem = None

                if subtask_elem is not None:
                    pop_submodule_tasks.append(subtask_elem)
                else:
                    break

            # 3. push into local map
            for i in range(len(pop_submodule_tasks)):
                # Check instance
                inst_i: SubModuleMovementRequest = pop_submodule_tasks[i]
                if inst_i.submodule_name not in local_submodule_task_queue:
                    continue

                # Init the submodule task status
                if len(local_submodule_task_queue[inst_i.submodule_name]) == 0:
                    local_submodule_task_state[inst_i.submodule_name] = SubTaskState(
                        idx_in_current_move=0
                    )

                # Push the request
                local_submodule_task_queue[inst_i.submodule_name].append(inst_i)

            # 4. reset the submodule task state
            next_full_jps = copy.deepcopy(current_full_jps)

            for submodule_name in local_submodule_task_queue.keys():
                # Extract the movement of submodule
                assert (
                    submodule_name in local_submodule_task_queue
                ), "Submodule name not found in local task queue."
                submodule_current_move: Optional[SubModuleMovementRequest] = None
                if len(local_submodule_task_queue[submodule_name]) > 0:
                    submodule_current_move = local_submodule_task_queue[submodule_name][
                        0
                    ]

                # Continue if submodule has no current move
                if submodule_current_move is None:
                    continue

                # Update jps for current module
                submodule_task_config = self._sub_task_map[submodule_name]
                idx_in_current_submodule_move = local_submodule_task_state[
                    submodule_name
                ].idx_in_current_move
                assert (
                    idx_in_current_submodule_move >= 0
                ), "Submodule task index is invalid."
                if idx_in_current_submodule_move < len(
                    submodule_current_move.motion_target
                ):
                    # Get jps
                    current_jps_submodule = submodule_current_move.motion_target[
                        idx_in_current_submodule_move
                    ]

                    # Update jps
                    for k in range(submodule_task_config.n_dof):
                        next_full_jps[submodule_task_config.dof_begin + k] = (
                            current_jps_submodule[k]
                        )

                    # Update index
                    next_idx_in_current_submodule_move = (
                        idx_in_current_submodule_move + 1
                    )
                    local_submodule_task_state[submodule_name] = SubTaskState(
                        idx_in_current_move=next_idx_in_current_submodule_move
                    )
                else:
                    # Pop this one from front
                    local_submodule_task_queue[submodule_name].pop(0)
                    if len(local_submodule_task_queue[submodule_name]) > 0:
                        local_submodule_task_state[submodule_name] = SubTaskState(
                            idx_in_current_move=0
                        )
                    else:
                        local_submodule_task_state[submodule_name] = SubTaskState(
                            idx_in_current_move=-1
                        )

            # 5. Reset of main task state
            if current_move is not None:
                assert idx_in_current_move >= 0 and next_endpoint2check >= 0
                if idx_in_current_move < len(current_move.motion_target):
                    # Update joint positions for this step
                    current_jps_main = current_move.motion_target[idx_in_current_move]

                    # Update jps
                    for k in range(self._main_task.n_dof):
                        next_full_jps[self._main_task.dof_begin + k] = current_jps_main[
                            k
                        ]

                    # Check if this point reaches a segment endpoint
                    if (
                        idx_in_current_move + 1
                        >= current_move.endpoint_index[next_endpoint2check]
                    ):
                        assert next_finished_move_id >= finished_move_id
                        finished_move_id = next_finished_move_id
                        next_endpoint2check += 1
                        next_finished_move_id += 1

                    self._on_robot_state_bg_thread(next_full_jps, finished_move_id)

                    # Move to next point in trajectory
                    idx_in_current_move += 1

                else:
                    # Movement finished, reset state for next movement
                    idx_in_current_move = -1
                    current_move = None
            # 6. Update for submodule only move
            elif submodule_current_move is not None:
                self._on_robot_state_bg_thread(next_full_jps, finished_move_id)
                submodule_current_move = None

            # Update shared state (non-blocking)
            current_full_jps = next_full_jps
            if self._lock.acquire(blocking=False):
                self._current_full_q = next_full_jps
                self._finished_move_id = finished_move_id
                self._lock.release()

            # Sleep
            time.sleep(self._interval)
