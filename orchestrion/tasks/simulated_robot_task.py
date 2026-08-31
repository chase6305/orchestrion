"""Deterministic in-process robot backend for demos and integration tests."""

import copy
import queue
import threading
import time
from typing import Callable, List, Optional

from orchestrion.utils.logger import logger

from .reduced_robot_task_interface import ReducedRobotTaskInterface


class SimulatedRobotTask(ReducedRobotTaskInterface):
    """Execute joint trajectories in real time without robot hardware."""

    def __init__(self, initial_joints: List[float]):
        if not initial_joints:
            raise ValueError("initial_joints must not be empty")
        if not self._is_finite_trajectory([initial_joints]):
            raise ValueError("initial_joints must contain finite real numbers")
        self._state = self.ReducedRobotState(initial_joints.copy(), -1, -1)
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._state_condition = threading.Condition(self._lock)
        self._queue: queue.SimpleQueue[ReducedRobotTaskInterface.MovementRequest] = (
            queue.SimpleQueue()
        )
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_change_callback: Optional[Callable[[], None]] = None

    def set_state_change_callback(self, callback: Optional[Callable[[], None]]) -> None:
        with self._lock:
            self._state_change_callback = callback

    def initialize(self) -> None:
        with self._lifecycle_lock:
            self._initialize_locked()

    def _initialize_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._wake_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="orchestrion-simulated-robot"
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lifecycle_lock:
            self._stop_locked(timeout)

    def _stop_locked(self, timeout: float) -> None:
        if not self._is_finite_real(timeout) or timeout < 0:
            raise ValueError("timeout must be non-negative and finite")
        self._stop_event.set()
        self._wake_event.set()
        with self._state_condition:
            self._state_condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("Simulated robot did not stop in time")
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def move_joint_trajectory_async(
        self,
        motion_target: List[List[float]],
        interval: float = 0.01,
        endpoint_index: Optional[List[int]] = None,
    ) -> int:
        if not self._is_finite_real(interval) or interval <= 0:
            return -1
        if not motion_target:
            return -1
        with self._lock:
            n_dof = len(self._state.jps)
        if any(len(point) != n_dof for point in motion_target):
            return -1
        if not self._is_finite_trajectory(motion_target):
            return -1
        endpoints = [len(motion_target)] if endpoint_index is None else endpoint_index
        if not self._check_assigned_endpoint_index(motion_target, endpoints):
            return -1

        with self._lock:
            move_id_begin = self._state.latest_sent_id + 1
            self._state.latest_sent_id += len(endpoints)
        self._queue.put(
            self.MovementRequest(
                motion_target=copy.deepcopy(motion_target),
                interval=interval,
                motion_id_begin=move_id_begin,
                endpoint_index=endpoints.copy(),
            )
        )
        self._notify_state_change()
        self._wake_event.set()
        return move_id_begin

    def query_state(self) -> ReducedRobotTaskInterface.ReducedRobotState:
        with self._lock:
            return copy.deepcopy(self._state)

    def wait_move(self, time_out: float = -1, interval: float = 0.05) -> bool:
        if not self._is_finite_real(time_out):
            raise ValueError("time_out must be finite")
        if not self._is_finite_real(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        with self._lock:
            target_id = self._state.latest_sent_id
        deadline = None if time_out < 0 else time.monotonic() + time_out
        with self._state_condition:
            while self._state.latest_finished_id < target_id:
                if self._stop_event.is_set():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                wait_interval = interval if remaining is None else min(interval, remaining)
                self._state_condition.wait(wait_interval)
            return True

    def _notify_state_change(self) -> None:
        with self._lock:
            callback = self._state_change_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.exception("Robot state-change callback failed")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait()
            self._wake_event.clear()
            while not self._stop_event.is_set():
                try:
                    request = self._queue.get_nowait()
                except queue.Empty:
                    break

                endpoint_position = 0
                finished_id = request.motion_id_begin
                for index, joints in enumerate(request.motion_target, start=1):
                    if self._stop_event.wait(request.interval):
                        return
                    if index >= request.endpoint_index[endpoint_position]:
                        latest_finished_id = finished_id
                        finished_id += 1
                        endpoint_position += 1
                    else:
                        with self._lock:
                            latest_finished_id = self._state.latest_finished_id
                    with self._state_condition:
                        self._state.jps = joints.copy()
                        self._state.latest_finished_id = latest_finished_id
                        self._state_condition.notify_all()
                    self._notify_state_change()
