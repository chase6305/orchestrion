import copy
import queue
import threading
import time
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

from orchestrion.tasks.reduced_robot_task_interface import ReducedRobotTaskInterface
from orchestrion.utils.logger import logger


class SubModuleTask:

    def __init__(self, name: str = "", dof_begin: int = 0, n_dof: int = 0):
        """Describe a named slice of the robot's joint vector."""
        self.name: str = name
        self.dof_begin: int = dof_begin
        self.n_dof: int = n_dof


@dataclass
class SubModuleMovementRequest:
    """A queued trajectory for one independently scheduled component."""

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


@dataclass
class _SubmoduleSchedule:
    """Worker-owned queue and cursor for one independent joint group."""

    moves: Deque[SubModuleMovementRequest] = field(default_factory=deque)
    point_index: int = 0
    next_step_at: float = 0.0


class ModularReducedRobotTask(ReducedRobotTaskInterface):
    """Schedule a main joint group and independent submodule trajectories."""

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
        self._lifecycle_lock = threading.RLock()
        self._state_condition = threading.Condition(self._lock)
        self._current_full_q = list(init_full_q)
        self._finished_move_id = -1
        self._latest_sent_id = -1
        self._cancelled_move_ids = set()
        self._motion_epoch = 0
        self._interval = interval

        self._main_task_queue = queue.SimpleQueue()
        self._main_task = copy.copy(main_task)
        self._bg_thread = None
        self._stop_event = threading.Event()

        self._sub_task_queue = queue.SimpleQueue()
        self._sub_task_map: Dict[str, SubModuleTask] = dict()
        for submodule in submodule_tasks:
            if submodule.name in self._sub_task_map:
                raise ValueError("Duplicate submodule name: {}".format(submodule.name))
            self._validate_module(submodule, len(init_full_q))
            self._sub_task_map[submodule.name] = copy.copy(submodule)
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
        self._cancelled_submodule_moves = {name: set() for name in self._sub_task_map}
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
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
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
        """Start the motion-processing worker."""
        with self._lifecycle_lock:
            self._initialize_locked()

    def _initialize_locked(self) -> None:
        if self._bg_thread is not None and self._bg_thread.is_alive():
            if self._stop_event.is_set():
                raise RuntimeError("Robot background thread is still stopping")
            logger.warning("Background thread is already running.")
            return

        self._stop_event.clear()
        self._bg_thread = threading.Thread(
            target=self._bg_thread_loop, name="orchestrion-robot-task"
        )
        try:
            self._bg_thread.start()
        except Exception:
            self._bg_thread = None
            self._stop_event.set()
            raise

    def stop(self, timeout: float = 5.0):
        """Cancel unfinished motion and wait for the worker to stop."""
        if threading.current_thread() is self._bg_thread:
            self._stop_locked(timeout)
            return
        with self._lifecycle_lock:
            self._stop_locked(timeout)

    def _stop_locked(self, timeout: float) -> None:
        if not self._is_finite_real(timeout) or timeout < 0:
            raise ValueError("timeout must be non-negative and finite")
        with self._state_condition:
            was_stopped = self._stop_event.is_set()
            self._stop_event.set()
            self._motion_epoch += 1
            self._cancelled_move_ids.update(
                range(self._finished_move_id + 1, self._latest_sent_id + 1)
            )
            for name in self._sub_task_map:
                self._cancelled_submodule_moves[name].update(
                    range(
                        self._submodule_finished_ids[name] + 1,
                        self._submodule_sent_ids[name] + 1,
                    )
                )
                self._submodule_active_ids[name] = None
            self._state_condition.notify_all()
        if not was_stopped:
            self._notify_state_change()
        if (
            self._bg_thread is not None
            and self._bg_thread is not threading.current_thread()
            and self._bg_thread.is_alive()
        ):
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
        """Enqueue a trajectory; return its first move ID or -1 on rejection."""
        if not self._is_finite_real(interval) or interval <= 0:
            return -1
        if not isinstance(motion_target, (list, tuple)) or not motion_target:
            return -1
        with self._lock:
            submission_epoch = self._motion_epoch
        motion_target = copy.deepcopy(motion_target)
        segment_endpoints = (
            [len(motion_target)]
            if endpoint_index is None
            else copy.deepcopy(endpoint_index)
        )
        if not self._check_assigned_endpoint_index(
            motion_target, segment_endpoints
        ):
            return -1

        if not motion_target or not self._is_finite_trajectory(motion_target):
            return -1
        if any(len(point) != self._main_task.n_dof for point in motion_target):
            return -1
        with self._lock:
            if self._stop_event.is_set() or submission_epoch != self._motion_epoch:
                return -1
            first_move_id = self._latest_sent_id + 1
            request = self.MovementRequest(
                motion_target=motion_target,
                interval=interval,
                motion_id_begin=first_move_id,
                endpoint_index=segment_endpoints,
            )
            self._main_task_queue.put(request)
            self._latest_sent_id += len(segment_endpoints)
            self._state_condition.notify_all()
        self._notify_state_change()
        return first_move_id

    def move_submodule_async(
        self,
        submodule_name: str,
        motion_target: List[List[float]],
        interval: float = 0.01,
    ) -> bool:
        """Compatibility wrapper returning whether a submodule move was accepted."""
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
        if (
            module is None
            or not isinstance(motion_target, (list, tuple))
            or not motion_target
        ):
            return -1
        with self._lock:
            submission_epoch = self._motion_epoch
        motion_target = copy.deepcopy(motion_target)
        if not motion_target or not self._is_finite_trajectory(motion_target):
            return -1
        if any(len(point) != module.n_dof for point in motion_target):
            return -1
        with self._state_condition:
            if self._stop_event.is_set() or submission_epoch != self._motion_epoch:
                return -1
            move_id = self._submodule_sent_ids[submodule_name] + 1
            self._submodule_sent_ids[submodule_name] = move_id
            self._sub_task_queue.put(
                SubModuleMovementRequest(
                    submodule_name=submodule_name,
                    motion_target=motion_target,
                    interval=interval,
                    move_id=move_id,
                )
            )
            self._state_condition.notify_all()
        self._notify_state_change()
        return move_id

    def move_submodules_trajectories_async(
        self,
        motion_targets: Dict[str, List[List[float]]],
        interval: float = 0.01,
    ) -> Dict[str, int]:
        """Atomically validate and enqueue trajectories for several submodules."""
        if not isinstance(motion_targets, dict) or not motion_targets:
            return {}
        if not self._is_finite_real(interval) or interval <= 0:
            return {}
        with self._lock:
            submission_epoch = self._motion_epoch
        # Snapshot the whole batch before assigning any move IDs. This preserves
        # all-or-nothing submission even if a caller mutates its input concurrently
        # or a custom container fails during copying.
        targets = copy.deepcopy(motion_targets)
        for submodule_name, motion_target in targets.items():
            module = self._sub_task_map.get(submodule_name)
            if (
                module is None
                or not isinstance(motion_target, list)
                or not motion_target
                or any(not isinstance(point, list) for point in motion_target)
            ):
                return {}
            if any(len(point) != module.n_dof for point in motion_target):
                return {}
            if not self._is_finite_trajectory(motion_target):
                return {}

        with self._state_condition:
            if self._stop_event.is_set() or submission_epoch != self._motion_epoch:
                return {}
            move_ids = {
                submodule_name: self._submodule_sent_ids[submodule_name] + 1
                for submodule_name in targets
            }
            for submodule_name, motion_target in targets.items():
                move_id = move_ids[submodule_name]
                self._submodule_sent_ids[submodule_name] = move_id
                self._sub_task_queue.put(
                    SubModuleMovementRequest(
                        submodule_name=submodule_name,
                        motion_target=motion_target,
                        interval=interval,
                        move_id=move_id,
                    )
                )
            self._state_condition.notify_all()
        self._notify_state_change()
        return move_ids

    def query_submodule_state(self, submodule_name: str) -> SubModuleState:
        module = self._sub_task_map.get(submodule_name)
        if module is None:
            raise KeyError("Unknown submodule: {}".format(submodule_name))
        with self._lock:
            begin = module.dof_begin
            return SubModuleState(
                name=submodule_name,
                positions=self._current_full_q[begin : begin + module.n_dof],
                latest_sent_id=self._submodule_sent_ids[submodule_name],
                latest_finished_id=self._submodule_finished_ids[submodule_name],
                active_move_id=self._submodule_active_ids[submodule_name],
                cancelled_move_ids=tuple(
                    sorted(self._cancelled_submodule_moves[submodule_name])
                ),
            )

    def query_submodule_states(
        self, submodule_names: Optional[List[str]] = None
    ) -> Dict[str, SubModuleState]:
        """Return an atomic snapshot of several or all submodule states."""
        if submodule_names is None:
            names = tuple(self._sub_task_map)
        else:
            if not isinstance(submodule_names, (list, tuple)) or any(
                not isinstance(name, str) or not name for name in submodule_names
            ):
                raise TypeError("submodule_names must be a list of names or None")
            names = tuple(submodule_names)
        with self._lock:
            for name in names:
                if name not in self._sub_task_map:
                    raise KeyError("Unknown submodule: {}".format(name))
            return {name: self.query_submodule_state(name) for name in names}

    def wait_submodule_move(
        self, submodule_name: str, move_id: int, timeout: Optional[float] = None
    ) -> bool:
        self._validate_submodule_move_id(move_id)
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
            while True:
                if move_id in self._cancelled_submodule_moves[submodule_name]:
                    return False
                if self._submodule_finished_ids[submodule_name] >= move_id:
                    return True
                if self._stop_event.is_set():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._state_condition.wait(remaining)

    def wait_submodule_moves(
        self, moves: Dict[str, int], timeout: Optional[float] = None
    ) -> bool:
        """Wait for several submodule moves using one shared timeout budget."""
        if not isinstance(moves, dict):
            raise TypeError("moves must be a dictionary")
        if timeout is not None and (
            not self._is_finite_real(timeout) or timeout < 0
        ):
            raise ValueError("timeout must be non-negative and finite, or None")
        requested = tuple(moves.items())
        for submodule_name, move_id in requested:
            if not isinstance(submodule_name, str) or not submodule_name:
                raise TypeError("moves must map submodule names to move IDs")
            self._validate_submodule_move_id(move_id)

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state_condition:
            for submodule_name, move_id in requested:
                if submodule_name not in self._sub_task_map:
                    raise KeyError("Unknown submodule: {}".format(submodule_name))
                if move_id > self._submodule_sent_ids[submodule_name]:
                    raise KeyError(
                        "Unknown move {} for submodule {}".format(
                            move_id, submodule_name
                        )
                    )
            while True:
                if any(
                    move_id in self._cancelled_submodule_moves[submodule_name]
                    for submodule_name, move_id in requested
                ):
                    return False
                if all(
                    self._submodule_finished_ids[submodule_name] >= move_id
                    for submodule_name, move_id in requested
                ):
                    return True
                if self._stop_event.is_set():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._state_condition.wait(remaining)

    @staticmethod
    def _validate_submodule_move_id(move_id: int) -> None:
        if isinstance(move_id, bool) or not isinstance(move_id, int):
            raise TypeError("move_id must be an integer")
        if move_id < 0:
            raise ValueError("move_id must be non-negative")

    def cancel_submodule_move(self, submodule_name: str, move_id: int) -> bool:
        self._validate_submodule_move_id(move_id)
        with self._state_condition:
            if submodule_name not in self._sub_task_map:
                raise KeyError("Unknown submodule: {}".format(submodule_name))
            if move_id in self._cancelled_submodule_moves[submodule_name]:
                return False
            if move_id > self._submodule_sent_ids[submodule_name]:
                return False
            if move_id <= self._submodule_finished_ids[submodule_name]:
                return False
            self._cancelled_submodule_moves[submodule_name].add(move_id)
            if self._submodule_active_ids[submodule_name] == move_id:
                self._submodule_active_ids[submodule_name] = None
            self._state_condition.notify_all()
        self._notify_state_change()
        return True

    def cancel_submodule_moves(self, moves: Dict[str, int]) -> Dict[str, bool]:
        """Cancel several submodule moves under one state lock."""
        if not isinstance(moves, dict):
            raise TypeError("moves must be a dictionary")
        requested = tuple(moves.items())
        for submodule_name, move_id in requested:
            if not isinstance(submodule_name, str) or not submodule_name:
                raise TypeError("moves must map submodule names to move IDs")
            self._validate_submodule_move_id(move_id)
        with self._state_condition:
            for submodule_name, _ in requested:
                if submodule_name not in self._sub_task_map:
                    raise KeyError("Unknown submodule: {}".format(submodule_name))
            results = {}
            for submodule_name, move_id in requested:
                cancelled_moves = self._cancelled_submodule_moves[submodule_name]
                can_cancel = (
                    move_id not in cancelled_moves
                    and move_id <= self._submodule_sent_ids[submodule_name]
                    and move_id > self._submodule_finished_ids[submodule_name]
                )
                results[submodule_name] = can_cancel
                if can_cancel:
                    cancelled_moves.add(move_id)
                    if self._submodule_active_ids[submodule_name] == move_id:
                        self._submodule_active_ids[submodule_name] = None
            changed = any(results.values())
            if changed:
                self._state_condition.notify_all()
        if changed:
            self._notify_state_change()
        return results

    def query_state(self) -> Optional[ReducedRobotTaskInterface.ReducedRobotState]:
        """Return a snapshot of joints and main movement IDs."""
        with self._lock:
            return self.ReducedRobotState(
                jps=self._current_full_q.copy(),
                latest_finished_id=self._finished_move_id,
                latest_sent_id=self._latest_sent_id,
                cancelled_move_ids=tuple(sorted(self._cancelled_move_ids)),
            )

    def wait_move(self, time_out: float = -1, interval: float = 0.05) -> bool:
        """Wait for the latest main move; negative time_out means no deadline."""
        if not self._is_finite_real(time_out):
            raise ValueError("time_out must be finite")
        if not self._is_finite_real(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        with self._lock:
            target_move_id = self._latest_sent_id
        deadline = None if time_out < 0 else time.monotonic() + time_out
        with self._state_condition:
            while True:
                if target_move_id in self._cancelled_move_ids:
                    return False
                if self._finished_move_id >= target_move_id:
                    return True
                if self._stop_event.is_set():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                wait_interval = (
                    interval if remaining is None else min(interval, remaining)
                )
                self._state_condition.wait(wait_interval)

    @abstractmethod
    def _on_robot_state_bg_thread(self, jps: List[float], jps_move_id: int):
        """Render or forward a joint update from the motion worker."""
        raise NotImplementedError

    @abstractmethod
    def _on_initial_state_bg_thread(self, jps: List[float]):
        """Publish the initial joints when the motion worker starts."""
        raise NotImplementedError

    def _admit_submodule_moves(
        self, schedules: Dict[str, _SubmoduleSchedule]
    ) -> None:
        # The shared submission lock keeps a validated batch visible to all
        # components in the same tick. No intermediate request list is needed.
        with self._lock:
            while True:
                try:
                    request = self._sub_task_queue.get_nowait()
                except queue.Empty:
                    return
                if request is None:
                    return
                schedule = schedules.get(request.submodule_name)
                if (
                    schedule is None
                    or request.move_id
                    in self._cancelled_submodule_moves[request.submodule_name]
                ):
                    continue
                if not schedule.moves:
                    schedule.point_index = 0
                    schedule.next_step_at = time.monotonic()
                schedule.moves.append(request)

    def _advance_submodule_moves(
        self,
        schedules: Dict[str, _SubmoduleSchedule],
        next_joint_positions: List[float],
    ) -> Tuple[bool, bool]:
        """Stage due points and report joint updates and completed requests."""
        joints_updated = False
        move_completed = False
        for submodule_name, schedule in schedules.items():
            if not schedule.moves:
                continue
            request = schedule.moves[0]
            with self._lock:
                cancelled = (
                    request.move_id in self._cancelled_submodule_moves[submodule_name]
                )
                if not cancelled:
                    self._submodule_active_ids[submodule_name] = request.move_id
            if cancelled:
                schedule.moves.popleft()
                schedule.point_index = 0
                schedule.next_step_at = time.monotonic()
                continue
            if time.monotonic() < schedule.next_step_at:
                continue
            if schedule.point_index < len(request.motion_target):
                module = self._sub_task_map[submodule_name]
                begin = module.dof_begin
                next_joint_positions[begin : begin + module.n_dof] = (
                    request.motion_target[schedule.point_index]
                )
                schedule.point_index += 1
                schedule.next_step_at = time.monotonic() + max(
                    request.interval, self._interval
                )
                joints_updated = True
            else:
                # The final point is already published; keep its interval before
                # announcing completion.
                schedule.moves.popleft()
                schedule.point_index = 0
                schedule.next_step_at = time.monotonic()
                with self._state_condition:
                    self._submodule_finished_ids[submodule_name] = max(
                        self._submodule_finished_ids[submodule_name], request.move_id
                    )
                    self._submodule_active_ids[submodule_name] = None
                    self._state_condition.notify_all()
                move_completed = True
        return joints_updated, move_completed

    def _bg_thread_loop(self):
        """Advance due points and publish the combined main/submodule snapshot."""
        with self._lock:
            current_joint_positions = self._current_full_q.copy()
            finished_move_id = self._finished_move_id
        self._on_initial_state_bg_thread(current_joint_positions)

        current_move: Optional[ReducedRobotTaskInterface.MovementRequest] = None
        main_point_index = 0
        next_endpoint_index = 0
        next_main_step_at = 0.0
        schedules = {name: _SubmoduleSchedule() for name in self._sub_task_map}

        while not self._stop_event.is_set():
            if current_move is None:
                try:
                    current_move = self._main_task_queue.get_nowait()
                    main_point_index = 0
                    next_endpoint_index = 0
                    next_main_step_at = time.monotonic()
                except queue.Empty:
                    pass
            self._admit_submodule_moves(schedules)
            next_joint_positions = current_joint_positions.copy()
            submodule_updated, submodule_completed = self._advance_submodule_moves(
                schedules, next_joint_positions
            )

            main_updated = False
            if current_move is not None:
                if main_point_index >= len(current_move.motion_target):
                    current_move = None
                elif time.monotonic() >= next_main_step_at:
                    begin = self._main_task.dof_begin
                    next_joint_positions[begin : begin + self._main_task.n_dof] = (
                        current_move.motion_target[main_point_index]
                    )
                    # Endpoints count points from one; the cursor is zero-based.
                    if main_point_index + 1 == current_move.endpoint_index[
                        next_endpoint_index
                    ]:
                        finished_move_id = (
                            current_move.motion_id_begin + next_endpoint_index
                        )
                        next_endpoint_index += 1
                    main_point_index += 1
                    main_updated = True

            if main_updated or submodule_updated:
                self._on_robot_state_bg_thread(next_joint_positions, finished_move_id)
                if main_updated:
                    # Preserve the full point interval after the device callback.
                    next_main_step_at = time.monotonic() + max(
                        current_move.interval, self._interval
                    )

            current_joint_positions = next_joint_positions
            with self._state_condition:
                self._current_full_q = next_joint_positions
                self._finished_move_id = finished_move_id
                if main_updated or submodule_updated:
                    self._state_condition.notify_all()
            if main_updated or submodule_updated or submodule_completed:
                self._notify_state_change()

            if current_move is None and not any(
                schedule.moves for schedule in schedules.values()
            ):
                # Share the submission lock so the empty check cannot miss a wake.
                with self._state_condition:
                    if (
                        self._main_task_queue.empty()
                        and self._sub_task_queue.empty()
                        and not self._stop_event.is_set()
                    ):
                        self._state_condition.wait()
            else:
                self._stop_event.wait(self._interval)
