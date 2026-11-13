"""Deterministic in-process robot backend for demos and integration tests."""

import copy
import queue
import threading
import time
from bisect import bisect_left
from typing import Callable, List, Optional

from orchestrion.utils.errors import exception_message
from orchestrion.utils.logger import log_error

from .reduced_robot_task_interface import ReducedRobotTaskInterface


class SimulatedRobotTask(ReducedRobotTaskInterface):
    """Pace joint trajectories with interruptible delays, without robot hardware."""

    def __init__(self, initial_joints: List[float]):
        if not initial_joints:
            raise ValueError("initial_joints must not be empty")
        if not self._is_finite_trajectory([initial_joints]):
            raise ValueError("initial_joints must contain finite real numbers")
        self._state = self.ReducedRobotState(list(initial_joints), -1, -1)
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._state_condition = threading.Condition(self._lock)
        self._queue: queue.SimpleQueue[ReducedRobotTaskInterface.MovementRequest] = (
            queue.SimpleQueue()
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_change_callback: Optional[Callable[[], None]] = None
        self._motion_epoch = 0

    def set_state_change_callback(self, callback: Optional[Callable[[], None]]) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        with self._lock:
            self._state_change_callback = callback

    def initialize(self) -> None:
        with self._lifecycle_lock:
            self._initialize_locked()

    def _initialize_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            if self._stop_event.is_set():
                raise RuntimeError("Simulated robot thread is still stopping")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="orchestrion-simulated-robot"
        )
        try:
            self._thread.start()
        except Exception:
            self._thread = None
            self._stop_event.set()
            raise

    def stop(self, timeout: float = 5.0) -> None:
        if threading.current_thread() is self._thread:
            # A state observer may request shutdown from the worker. It must not
            # wait for a lifecycle lock held by another caller joining this worker.
            self._stop_locked(timeout)
            return
        with self._lifecycle_lock:
            self._stop_locked(timeout)

    def _stop_locked(self, timeout: float) -> None:
        if not self._is_finite_real(timeout) or timeout < 0:
            raise ValueError("timeout must be non-negative and finite")
        if timeout > threading.TIMEOUT_MAX:
            raise ValueError("timeout must not exceed threading.TIMEOUT_MAX")
        timeout = float(timeout)
        with self._state_condition:
            was_stopped = self._stop_event.is_set()
            self._stop_event.set()
            self._motion_epoch += 1
            unfinished_move_ids = range(
                self._state.latest_finished_id + 1,
                self._state.latest_sent_id + 1,
            )
            self._state.cancelled_move_ids = tuple(
                sorted(set(self._state.cancelled_move_ids).union(unfinished_move_ids))
            )
            self._state_condition.notify_all()
        # Admission is closed. Discard queued work before callbacks or joins can
        # fail, so a later explicit restart cannot replay cancelled trajectories.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if not was_stopped:
            self._notify_state_change()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("Simulated robot did not stop in time")

    def move_joint_trajectory_async(
        self,
        motion_target: List[List[float]],
        interval: float = 0.01,
        endpoint_index: Optional[List[int]] = None,
    ) -> int:
        if not self._is_positive_finite_real(interval):
            return -1
        if interval > threading.TIMEOUT_MAX:
            return -1
        interval = float(interval)
        if not isinstance(motion_target, (list, tuple)) or not motion_target:
            return -1
        with self._lock:
            expected_joint_count = len(self._state.jps)
            submission_epoch = self._motion_epoch
        motion_target = self._copy_motion_input(motion_target)
        if not self._is_finite_trajectory(motion_target):
            return -1
        if any(len(point) != expected_joint_count for point in motion_target):
            return -1
        segment_endpoints = (
            [len(motion_target)]
            if endpoint_index is None
            else self._copy_motion_input(endpoint_index)
        )
        if not self._check_assigned_endpoint_index(motion_target, segment_endpoints):
            return -1

        with self._lock:
            if self._stop_event.is_set() or submission_epoch != self._motion_epoch:
                return -1
            first_move_id = self._state.latest_sent_id + 1
            request = self.MovementRequest(
                motion_target=motion_target,
                interval=interval,
                motion_id_begin=first_move_id,
                endpoint_index=segment_endpoints,
            )
            self._queue.put(request)
            self._state.latest_sent_id += len(segment_endpoints)
            self._state_condition.notify_all()
        self._notify_state_change()
        return first_move_id

    def query_state(self) -> ReducedRobotTaskInterface.ReducedRobotState:
        with self._lock:
            # Integer history is immutable; copy the remaining state graph normally.
            cancelled_move_ids = self._state.cancelled_move_ids
            return copy.deepcopy(
                self._state, memo={id(cancelled_move_ids): cancelled_move_ids}
            )

    def wait_move(self, time_out: float = -1, interval: float = 0.05) -> bool:
        if not self._is_finite_real(time_out):
            raise ValueError("time_out must be finite")
        if not self._is_positive_finite_real(interval):
            raise ValueError("interval must be positive and finite")
        interval = min(float(interval), threading.TIMEOUT_MAX)
        with self._lock:
            target_move_id = self._state.latest_sent_id
        deadline = None if time_out < 0 else time.monotonic() + time_out
        with self._state_condition:
            while True:
                # stop() publishes sorted IDs; avoid rescanning on each wake.
                cancelled_move_ids = self._state.cancelled_move_ids
                cancelled_move_index = bisect_left(cancelled_move_ids, target_move_id)
                if (
                    cancelled_move_index < len(cancelled_move_ids)
                    and cancelled_move_ids[cancelled_move_index] == target_move_id
                ):
                    return False
                if self._state.latest_finished_id >= target_move_id:
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

    def _notify_state_change(self) -> None:
        with self._lock:
            callback = self._state_change_callback
        if callback is not None:
            try:
                callback()
            except Exception as exc:
                log_error(
                    "Robot state-change callback failed: %s", exception_message(exc)
                )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._run_next_trajectory()
        except Exception as exc:
            log_error(
                "Simulated robot motion worker failed: %s", exception_message(exc)
            )
            self._stop_locked(0.0)

    def _run_next_trajectory(self) -> None:
        """Release the request and its point aliases before the next idle wait."""
        with self._state_condition:
            # Submission and the empty check share this lock, so a producer
            # cannot notify between checking the queue and going to sleep.
            while self._queue.empty() and not self._stop_event.is_set():
                self._state_condition.wait()
            if self._stop_event.is_set():
                return
            request = self._queue.get_nowait()

        next_endpoint_index = 0
        next_finished_move_id = request.motion_id_begin
        for point_number, joint_positions in enumerate(request.motion_target, start=1):
            if self._stop_event.wait(request.interval):
                return
            with self._state_condition:
                self._state.jps = list(joint_positions)
                # Endpoints count points from one, matching point_number.
                if point_number == request.endpoint_index[next_endpoint_index]:
                    self._state.latest_finished_id = next_finished_move_id
                    next_finished_move_id += 1
                    next_endpoint_index += 1
                self._state_condition.notify_all()
            self._notify_state_change()
