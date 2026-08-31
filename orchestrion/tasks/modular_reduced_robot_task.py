import copy
import queue
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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
    move_id: int = -1


@dataclass(frozen=True)
class SubModuleState:
    name: str
    positions: List[float]
    latest_sent_id: int = -1
    latest_finished_id: int = -1
    active_move_id: Optional[int] = None
    cancelled_move_ids: tuple = ()


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
        if not self._is_finite_real(interval) or interval <= 0:
            raise ValueError("Scheduler interval must be positive and finite")
        if not init_full_q or not self._is_finite_trajectory([init_full_q]):
            raise ValueError("Initial joints must be finite real numbers")
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._state_condition = threading.Condition(self._lock)
        self._current_full_q = init_full_q.copy()
        self._finished_move_id = -1
        self._latest_sent_id = -1
        self._interval = interval

        # Queues and threads
        self._main_task_queue = queue.SimpleQueue()
        self._main_task = main_task
        self._bg_thread = None
        self._stop_event = threading.Event()

        # Sub-module management
        self._sub_task_queue = queue.SimpleQueue()
        self._sub_task_map: Dict[str, SubModuleTask] = dict()
        for elem in submodule_tasks:
            if elem.name in self._sub_task_map:
                raise ValueError("Duplicate submodule name: {}".format(elem.name))
            self._validate_module(elem, len(init_full_q))
            self._sub_task_map[elem.name] = elem
        self._validate_module(main_task, len(init_full_q))
        if main_task.name in self._sub_task_map:
            raise ValueError("Main task and submodule names must be unique")
        modules = [main_task, *submodule_tasks]
        for index, module in enumerate(modules):
            module_range = range(module.dof_begin, module.dof_begin + module.n_dof)
            for other in modules[index + 1 :]:
                other_range = range(other.dof_begin, other.dof_begin + other.n_dof)
                overlaps = (
                    module_range.start < other_range.stop
                    and other_range.start < module_range.stop
                )
                if overlaps:
                    raise ValueError(
                        "Module DOF ranges overlap: {} and {}".format(
                            module.name, other.name
                        )
                    )
        self._submodule_sent_ids = {name: -1 for name in self._sub_task_map}
        self._submodule_finished_ids = {name: -1 for name in self._sub_task_map}
        self._submodule_active_ids = {name: None for name in self._sub_task_map}
        self._cancelled_submodule_moves = set()
        self._state_change_callback: Optional[Callable[[], None]] = None

    @staticmethod
    def _validate_module(module: SubModuleTask, total_dof: int) -> None:
        if not isinstance(module.name, str) or not module.name:
            raise ValueError("Module name must be a non-empty string")
        if (
            isinstance(module.dof_begin, bool)
            or not isinstance(module.dof_begin, int)
            or isinstance(module.n_dof, bool)
            or not isinstance(module.n_dof, int)
        ):
            raise TypeError("Module DOF range must use integers")
        if module.dof_begin < 0 or module.n_dof <= 0:
            raise ValueError("Module DOF range must be positive")
        if module.dof_begin + module.n_dof > total_dof:
            raise ValueError("Module DOF range exceeds the robot state")

    @property
    def scheduler_interval(self) -> float:
        return self._interval

    def set_state_change_callback(self, callback: Optional[Callable[[], None]]) -> None:
        with self._lock:
            self._state_change_callback = callback

    def _notify_state_change(self) -> None:
        with self._lock:
            callback = self._state_change_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.exception("Robot state-change callback failed")

    def initialize(self):
        """
        Initialize and start the background thread for processing movement requests.
        """
        with self._lifecycle_lock:
            self._initialize_locked()

    def _initialize_locked(self) -> None:
        if self._bg_thread is not None and self._bg_thread.is_alive():
            logger.warning("Background thread is already running.")
            return

        self._stop_event.clear()
        self._bg_thread = threading.Thread(
            target=self._bg_thread_loop, name="orchestrion-robot-task"
        )
        self._bg_thread.start()

    def stop(self, timeout: float = 5.0):
        """
        Stop the background thread and wait for it to finish.
        """
        with self._lifecycle_lock:
            self._stop_locked(timeout)

    def _stop_locked(self, timeout: float) -> None:
        if not self._is_finite_real(timeout) or timeout < 0:
            raise ValueError("timeout must be non-negative and finite")
        self._stop_event.set()
        with self._state_condition:
            for name in self._sub_task_map:
                for move_id in range(
                    self._submodule_finished_ids[name] + 1,
                    self._submodule_sent_ids[name] + 1,
                ):
                    self._cancelled_submodule_moves.add((name, move_id))
                self._submodule_active_ids[name] = None
            self._state_condition.notify_all()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=timeout)
            if self._bg_thread.is_alive():
                raise TimeoutError("Robot background thread did not stop in time")
        self._drain_queue(self._main_task_queue)
        self._drain_queue(self._sub_task_queue)

    @staticmethod
    def _drain_queue(task_queue: queue.SimpleQueue) -> None:
        while True:
            try:
                task_queue.get_nowait()
            except queue.Empty:
                return

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
        if not self._is_finite_real(interval) or interval <= 0:
            return -1
        if not motion_target:
            return -1
        assigned_endpoint_idx = (
            [len(motion_target)] if endpoint_index is None else endpoint_index
        )
        if not self._check_assigned_endpoint_index(
            motion_target, assigned_endpoint_idx
        ):
            return -1

        if any(len(point) != self._main_task.n_dof for point in motion_target):
            return -1
        if not self._is_finite_trajectory(motion_target):
            return -1
        with self._lock:
            move_id_begin = self._latest_sent_id + 1
            request = self.MovementRequest(
                motion_target=copy.deepcopy(motion_target),
                interval=interval,
                motion_id_begin=move_id_begin,
                endpoint_index=assigned_endpoint_idx.copy(),
            )
            self._main_task_queue.put(request)
            self._latest_sent_id += len(assigned_endpoint_idx)
        return move_id_begin

    def move_submodule_async(
        self,
        submodule_name: str,
        motion_target: List[List[float]],
        interval: float = 0.01,
    ) -> bool:
        """
        Move a sub-module asynchronously.

        Args:
            submodule_name (str): Name of the sub-module to move.
            motion_target (List[List[float]]): Target joint positions for the sub-module.
            interval (float): Time interval between points.

        Returns:
            bool: True if the request was accepted, False otherwise.
        """
        return self.move_submodule_trajectory_async(
            submodule_name, motion_target, interval
        ) >= 0

    def move_submodule_trajectory_async(
        self,
        submodule_name: str,
        motion_target: List[List[float]],
        interval: float = 0.01,
    ) -> int:
        if not self._is_finite_real(interval) or interval <= 0:
            return -1
        module = self._sub_task_map.get(submodule_name)
        if module is None or not motion_target:
            return -1
        if any(len(point) != module.n_dof for point in motion_target):
            return -1
        if not self._is_finite_trajectory(motion_target):
            return -1
        with self._state_condition:
            move_id = self._submodule_sent_ids[submodule_name] + 1
            self._submodule_sent_ids[submodule_name] = move_id
            self._sub_task_queue.put(
                SubModuleMovementRequest(
                    submodule_name=submodule_name,
                    motion_target=copy.deepcopy(motion_target),
                    interval=interval,
                    move_id=move_id,
                )
            )
            self._state_condition.notify_all()
        return move_id

    def query_submodule_state(self, submodule_name: str) -> SubModuleState:
        module = self._sub_task_map.get(submodule_name)
        if module is None:
            raise KeyError("Unknown submodule: {}".format(submodule_name))
        with self._lock:
            begin = module.dof_begin
            return SubModuleState(
                name=submodule_name,
                positions=self._current_full_q[begin : begin + module.n_dof].copy(),
                latest_sent_id=self._submodule_sent_ids[submodule_name],
                latest_finished_id=self._submodule_finished_ids[submodule_name],
                active_move_id=self._submodule_active_ids[submodule_name],
                cancelled_move_ids=tuple(
                    sorted(
                        move_id
                        for name, move_id in self._cancelled_submodule_moves
                        if name == submodule_name
                    )
                ),
            )

    def wait_submodule_move(
        self, submodule_name: str, move_id: int, timeout: Optional[float] = None
    ) -> bool:
        if isinstance(move_id, bool) or not isinstance(move_id, int):
            raise TypeError("move_id must be an integer")
        if move_id < 0:
            raise ValueError("move_id must be non-negative")
        if timeout is not None and (
            not self._is_finite_real(timeout) or timeout < 0
        ):
            raise ValueError("timeout must be non-negative and finite, or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state_condition:
            if submodule_name not in self._sub_task_map:
                raise KeyError("Unknown submodule: {}".format(submodule_name))
            if move_id > self._submodule_sent_ids[submodule_name]:
                raise KeyError(
                    "Unknown move {} for submodule {}".format(
                        move_id, submodule_name
                    )
                )
            if (submodule_name, move_id) in self._cancelled_submodule_moves:
                return False
            while self._submodule_finished_ids[submodule_name] < move_id:
                if (submodule_name, move_id) in self._cancelled_submodule_moves:
                    return False
                if self._stop_event.is_set():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._state_condition.wait(remaining)
            return True

    def cancel_submodule_move(self, submodule_name: str, move_id: int) -> bool:
        if isinstance(move_id, bool) or not isinstance(move_id, int):
            raise TypeError("move_id must be an integer")
        if move_id < 0:
            raise ValueError("move_id must be non-negative")
        with self._state_condition:
            if submodule_name not in self._sub_task_map:
                raise KeyError("Unknown submodule: {}".format(submodule_name))
            if move_id > self._submodule_sent_ids[submodule_name]:
                return False
            if move_id <= self._submodule_finished_ids[submodule_name]:
                return False
            self._cancelled_submodule_moves.add((submodule_name, move_id))
            if self._submodule_active_ids[submodule_name] == move_id:
                self._submodule_active_ids[submodule_name] = None
            self._state_condition.notify_all()
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
                jps=self._current_full_q.copy(),
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
        if not self._is_finite_real(time_out):
            raise ValueError("time_out must be finite")
        if not self._is_finite_real(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        sent_id = self.query_state().latest_sent_id
        deadline = None if time_out < 0 else time.monotonic() + time_out
        with self._state_condition:
            while self._finished_move_id < sent_id:
                if self._stop_event.is_set():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                wait_interval = interval if remaining is None else min(interval, remaining)
                self._state_condition.wait(wait_interval)
            return True

    @abstractmethod
    def _on_robot_state_bg_thread(self, jps: List[float], jps_move_id: int):
        """
        Callback for updating the main robot state in the background thread.

        Args:
            jps (List[float]): Current joint positions.
            jps_move_id (int): Move ID associated with the joint positions.
        """
        raise NotImplementedError

    @abstractmethod
    def _on_initial_state_bg_thread(self, jps: List[float]):
        """
        Callback for initial state update of the main robot in the background thread.

        Args:
            jps (List[float]): Current joint positions.
        """
        raise NotImplementedError

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
        next_main_step_at: float = 0.0

        @dataclass
        class SubTaskState(object):
            idx_in_current_move: int = -1
            next_step_at: float = 0.0

        # Init for local queue
        local_submodule_task_queue: Dict[str, List[SubModuleMovementRequest]] = dict()
        local_submodule_task_state: Dict[str, SubTaskState] = dict()
        for task_name in self._sub_task_map.keys():
            local_submodule_task_queue[task_name] = list()
            local_submodule_task_state[task_name] = SubTaskState(idx_in_current_move=-1)
        submodule_current_move = None

        # Execution loop
        while not self._stop_event.is_set():
            # 1. If no current move, try to get a new move from the queue
            if current_move is None:
                try:
                    elem: self.MovementRequest = self._main_task_queue.get_nowait()
                    current_move = elem
                    idx_in_current_move = 0
                    next_endpoint2check = 0
                    next_finished_move_id = elem.motion_id_begin
                    next_main_step_at = time.monotonic()
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
                with self._lock:
                    if (
                        inst_i.submodule_name,
                        inst_i.move_id,
                    ) in self._cancelled_submodule_moves:
                        continue

                # Init the submodule task status
                if len(local_submodule_task_queue[inst_i.submodule_name]) == 0:
                    local_submodule_task_state[inst_i.submodule_name] = SubTaskState(
                        idx_in_current_move=0, next_step_at=time.monotonic()
                    )

                # Push the request
                local_submodule_task_queue[inst_i.submodule_name].append(inst_i)

            # 4. reset the submodule task state
            next_full_jps = copy.deepcopy(current_full_jps)
            submodule_updated = False

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
                with self._lock:
                    cancelled = (
                        (submodule_name, submodule_current_move.move_id)
                        in self._cancelled_submodule_moves
                    )
                if cancelled:
                    local_submodule_task_queue[submodule_name].pop(0)
                    local_submodule_task_state[submodule_name] = SubTaskState(
                        idx_in_current_move=(
                            0
                            if local_submodule_task_queue[submodule_name]
                            else -1
                        ),
                        next_step_at=time.monotonic(),
                    )
                    continue

                with self._lock:
                    self._submodule_active_ids[submodule_name] = (
                        submodule_current_move.move_id
                    )

                # Update jps for current module
                submodule_task_config = self._sub_task_map[submodule_name]
                idx_in_current_submodule_move = local_submodule_task_state[
                    submodule_name
                ].idx_in_current_move
                if (
                    time.monotonic()
                    < local_submodule_task_state[submodule_name].next_step_at
                ):
                    continue
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
                    submodule_updated = True

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
                        idx_in_current_move=next_idx_in_current_submodule_move,
                        next_step_at=time.monotonic()
                        + max(submodule_current_move.interval, self._interval),
                    )
                else:
                    # Pop this one from front
                    local_submodule_task_queue[submodule_name].pop(0)
                    with self._state_condition:
                        self._submodule_finished_ids[submodule_name] = max(
                            self._submodule_finished_ids[submodule_name],
                            submodule_current_move.move_id,
                        )
                        self._submodule_active_ids[submodule_name] = None
                        self._state_condition.notify_all()
                    if len(local_submodule_task_queue[submodule_name]) > 0:
                        local_submodule_task_state[submodule_name] = SubTaskState(
                            idx_in_current_move=0, next_step_at=time.monotonic()
                        )
                    else:
                        local_submodule_task_state[submodule_name] = SubTaskState(
                            idx_in_current_move=-1
                        )

            # 5. Reset of main task state
            main_updated = False
            if current_move is not None:
                assert idx_in_current_move >= 0 and next_endpoint2check >= 0
                if (
                    idx_in_current_move < len(current_move.motion_target)
                    and time.monotonic() >= next_main_step_at
                ):
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
                    main_updated = True

                    # Move to next point in trajectory
                    idx_in_current_move += 1
                    next_main_step_at = time.monotonic() + max(
                        current_move.interval, self._interval
                    )

                elif idx_in_current_move >= len(current_move.motion_target):
                    # Movement finished, reset state for next movement
                    idx_in_current_move = -1
                    current_move = None
            # 6. Update for submodule only move
            if submodule_updated and not main_updated:
                self._on_robot_state_bg_thread(next_full_jps, finished_move_id)

            # Update shared state (non-blocking)
            current_full_jps = next_full_jps
            with self._state_condition:
                self._current_full_q = next_full_jps
                self._finished_move_id = finished_move_id
                if main_updated or submodule_updated:
                    self._state_condition.notify_all()
            if main_updated or submodule_updated:
                self._notify_state_change()

            # Sleep
            self._stop_event.wait(self._interval)
