"""Robot and peripheral task orchestration."""

import concurrent.futures
import copy
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple

from orchestrion.move_sync_option import MoveSyncOption
from orchestrion.request import RequestResult, RequestStatus, TimelineEvent
from orchestrion.tasks.generic_task import GenericTask
from orchestrion.tasks.reduced_robot_task_interface import ReducedRobotTaskInterface
from orchestrion.utils.logger import logger
from orchestrion.utils.types import PeekResponseResultType


class TaskPilot:
    """Coordinate robot movement and optionally synchronized peripheral requests."""

    @dataclass(frozen=True)
    class BackgroundRequest:
        service_name: str
        request_id: int
        content: Optional[Dict] = None
        associated_move_id: int = -1

    def __init__(
        self,
        robot_task: ReducedRobotTaskInterface,
        task_map: Optional[Dict[str, GenericTask]] = None,
        executor_owned: Optional[concurrent.futures.ThreadPoolExecutor] = None,
        poll_interval: float = 0.05,
        timeline_capacity: int = 1000,
    ):
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, Real)
            or not math.isfinite(poll_interval)
            or poll_interval <= 0
        ):
            raise ValueError("poll_interval must be positive and finite")
        if isinstance(timeline_capacity, bool) or not isinstance(
            timeline_capacity, int
        ):
            raise TypeError("timeline_capacity must be an integer")
        if timeline_capacity <= 0:
            raise ValueError("timeline_capacity must be positive")
        self._robot_task = robot_task
        self._task_map = dict(task_map) if task_map is not None else {}
        if any(not isinstance(name, str) or not name for name in self._task_map):
            raise ValueError("Task service names must be non-empty strings")
        self._executor = executor_owned
        self._executor_shutdown = False
        self._poll_interval = poll_interval
        self._timeline_capacity = timeline_capacity

        self._task_queue: queue.SimpleQueue[TaskPilot.BackgroundRequest] = (
            queue.SimpleQueue()
        )
        self._next_request_id = 0
        self._state_condition = threading.Condition()
        self._requests: Dict[int, RequestResult] = {}
        self._request_waiters: Dict[int, int] = {}
        self._timeline: deque[TimelineEvent] = deque(maxlen=timeline_capacity)
        self._running_requests: Dict[int, TaskPilot.BackgroundRequest] = {}
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._initialized = False

    @property
    def task_map(self) -> Dict[str, GenericTask]:
        return self._task_map

    @property
    def robot_task(self) -> ReducedRobotTaskInterface:
        return self._robot_task

    def _rollback_initialization(self, initialized_tasks: List[GenericTask]) -> None:
        for task in reversed(initialized_tasks):
            try:
                task.stop()
            except Exception:
                logger.exception("Failed to roll back peripheral task")
        for task in self._task_map.values():
            set_completion_callback = getattr(task, "set_completion_callback", None)
            if callable(set_completion_callback):
                try:
                    set_completion_callback(None)
                except Exception:
                    logger.exception("Failed to clear task completion callback")
        set_state_callback = getattr(
            self._robot_task, "set_state_change_callback", None
        )
        if callable(set_state_callback):
            try:
                set_state_callback(None)
            except Exception:
                logger.exception("Failed to clear robot state callback")
        try:
            self._robot_task.stop()
        except Exception:
            logger.exception("Failed to roll back robot task")

    def initialize(self) -> None:
        if self._initialized:
            logger.warning("TaskPilot is already initialized.")
            return
        if self._executor_shutdown:
            raise RuntimeError(
                "TaskPilot cannot restart after its owned executor was shut down"
            )
        self._stop_event.clear()
        self._wake_event.clear()
        try:
            self._robot_task.initialize()
        except Exception:
            try:
                self._robot_task.stop()
            except Exception:
                logger.exception("Failed to roll back robot task initialization")
            raise
        initialized_tasks = []
        try:
            for task in self._task_map.values():
                set_completion_callback = getattr(task, "set_completion_callback", None)
                if callable(set_completion_callback):
                    set_completion_callback(self._wake_event.set)
                initialized_tasks.append(task)
                task.initialize(executor=self._executor)
        except Exception:
            self._rollback_initialization(initialized_tasks)
            raise

        try:
            set_callback = getattr(
                self._robot_task, "set_state_change_callback", None
            )
            if callable(set_callback):
                set_callback(self._wake_event.set)
            self._bg_thread = threading.Thread(
                target=self._bg_thread_loop, name="orchestrion-task-pilot"
            )
            self._bg_thread.start()
            self._initialized = True
        except Exception:
            self._rollback_initialization(initialized_tasks)
            self._bg_thread = None
            raise

    def stop(self, timeout: float = 5.0) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be non-negative and finite")
        if not self._initialized:
            logger.warning("TaskPilot is not running or already stopped.")
            return
        self._stop_event.set()
        self._wake_event.set()
        thread_timed_out = False
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=timeout)
            thread_timed_out = self._bg_thread.is_alive()

        self._cancel_nonterminal("TaskPilot stopped")
        task_stop_error: Optional[Exception] = None
        try:
            for task in self._task_map.values():
                try:
                    task.stop()
                except Exception as exc:
                    logger.exception("Failed to stop peripheral task")
                    if task_stop_error is None:
                        task_stop_error = exc
                finally:
                    set_completion_callback = getattr(
                        task, "set_completion_callback", None
                    )
                    if callable(set_completion_callback):
                        try:
                            set_completion_callback(None)
                        except Exception:
                            logger.exception("Failed to clear task completion callback")
        finally:
            try:
                if self._executor is not None:
                    # Running Python callbacks cannot be forcefully interrupted.
                    # Keep stop(timeout=...) bounded while still rejecting new
                    # submissions and cancelling work that has not started.
                    self._executor.shutdown(wait=False, cancel_futures=True)
                    self._executor_shutdown = True
            finally:
                set_callback = getattr(
                    self._robot_task, "set_state_change_callback", None
                )
                if callable(set_callback):
                    try:
                        set_callback(None)
                    except Exception:
                        logger.exception("Failed to clear robot state callback")
                try:
                    self._robot_task.stop()
                finally:
                    self._initialized = False
        if thread_timed_out:
            raise TimeoutError("TaskPilot background thread did not stop in time")
        if task_stop_error is not None:
            raise RuntimeError("A peripheral task failed to stop") from task_stop_error

    def call_srv_async(
        self,
        srv_name: str,
        content: Optional[Dict] = None,
        sync_option: Optional[MoveSyncOption] = None,
    ) -> int:
        if srv_name not in self._task_map:
            raise KeyError("Unknown service: {}".format(srv_name))
        if not self._initialized:
            raise RuntimeError("TaskPilot must be initialized before accepting requests")
        sync_option = sync_option or MoveSyncOption.sync_w_latest_move()

        associated_move_id = -1
        if sync_option.need_sync:
            if sync_option.associated_move_id >= 0:
                associated_move_id = sync_option.associated_move_id
            else:
                robot_state = self._robot_task.query_state()
                if robot_state is None:
                    raise RuntimeError("Robot state is unavailable")
                associated_move_id = robot_state.latest_sent_id

        request_content = copy.deepcopy(content)
        with self._state_condition:
            request_id = self._next_request_id
            self._next_request_id += 1
            now = time.time()
            self._requests[request_id] = RequestResult(
                request_id=request_id,
                service_name=srv_name,
                status=RequestStatus.QUEUED,
                associated_move_id=associated_move_id,
                created_at=now,
            )
            self._record_event_locked(request_id, RequestStatus.QUEUED)
            self._task_queue.put(
                self.BackgroundRequest(
                    service_name=srv_name,
                    request_id=request_id,
                    content=request_content,
                    associated_move_id=associated_move_id,
                )
            )
        self._wake_event.set()
        return request_id

    def query_request(self, request_id: int) -> RequestResult:
        with self._state_condition:
            try:
                return self._requests[request_id]
            except KeyError:
                raise KeyError("Unknown request: {}".format(request_id)) from None

    def wait_request(self, request_id: int, timeout: Optional[float] = None) -> RequestResult:
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be non-negative and finite, or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state_condition:
            if request_id not in self._requests:
                raise KeyError("Unknown request: {}".format(request_id))
            self._request_waiters[request_id] = (
                self._request_waiters.get(request_id, 0) + 1
            )
            try:
                while not self._requests[request_id].status.terminal:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError(
                            "Request {} did not finish in time".format(request_id)
                        )
                    self._state_condition.wait(remaining)
                return self._requests[request_id]
            finally:
                waiter_count = self._request_waiters[request_id] - 1
                if waiter_count:
                    self._request_waiters[request_id] = waiter_count
                else:
                    self._request_waiters.pop(request_id)

    def cancel_request(self, request_id: int) -> bool:
        with self._state_condition:
            result = self._requests.get(request_id)
            if result is None:
                raise KeyError("Unknown request: {}".format(request_id))
            if result.status.terminal:
                return False
            running = request_id in self._running_requests
            if running:
                cancel = getattr(
                    self._task_map[result.service_name], "cancel_request", None
                )
                if not callable(cancel) or not cancel(request_id):
                    return False
            # Keep the state lock across delegated cancellation and transition so
            # a completion callback cannot race this request into FAILED first.
            self._transition(
                request_id, RequestStatus.CANCELLED, error="Cancelled by user"
            )
        self._wake_event.set()
        return True

    def timeline(self, request_id: Optional[int] = None) -> List[TimelineEvent]:
        with self._state_condition:
            events = list(self._timeline)
        if request_id is None:
            return events
        return [event for event in events if event.request_id == request_id]

    def prune_completed_requests(self, keep_last: int = 1000) -> int:
        """Forget old terminal requests while preserving active requests.

        Timeline storage remains independently bounded by ``timeline_capacity``.
        Returns the number of request records removed from this pilot.
        """
        if isinstance(keep_last, bool) or not isinstance(keep_last, int):
            raise TypeError("keep_last must be an integer")
        if keep_last < 0:
            raise ValueError("keep_last must be non-negative")
        with self._state_condition:
            terminal_ids = [
                request_id
                for request_id, result in self._requests.items()
                if result.status.terminal
                and self._request_waiters.get(request_id, 0) == 0
            ]
            prune_ids = terminal_ids if keep_last == 0 else terminal_ids[:-keep_last]
            pruned = [
                (request_id, self._requests[request_id].service_name)
                for request_id in prune_ids
            ]
            for request_id, _ in pruned:
                self._requests.pop(request_id)

        for request_id, service_name in pruned:
            forget = getattr(self._task_map[service_name], "forget_response", None)
            if callable(forget):
                try:
                    forget(request_id)
                except Exception:
                    logger.exception("Failed to forget response %s", request_id)
        return len(pruned)

    def move_joint_trajectory_async(
        self,
        motion_target: List[List[float]],
        interval: float = 0.01,
        endpoint_index: Optional[List[int]] = None,
    ) -> Tuple[int, int]:
        n_segments = 1 if endpoint_index is None else len(endpoint_index)
        move_id_begin = self._robot_task.move_joint_trajectory_async(
            motion_target=motion_target,
            interval=interval,
            endpoint_index=endpoint_index,
        )
        if move_id_begin < 0:
            return move_id_begin, move_id_begin
        return move_id_begin, move_id_begin + n_segments

    def query_robot_state(self):
        return self._robot_task.query_state()

    def wait_move(self, time_out: float = -1, interval: float = 0.05) -> bool:
        return self._robot_task.wait_move(time_out=time_out, interval=interval)

    def _record_event_locked(
        self, request_id: int, status: RequestStatus, message: Optional[str] = None
    ) -> None:
        result = self._requests[request_id]
        self._timeline.append(
            TimelineEvent.now(request_id, result.service_name, status, message)
        )

    def _transition(
        self,
        request_id: int,
        status: RequestStatus,
        content: Any = None,
        error: Optional[str] = None,
    ) -> bool:
        with self._state_condition:
            current = self._requests.get(request_id)
            if current is None or current.status.terminal:
                return False
            now = time.time()
            self._requests[request_id] = replace(
                current,
                status=status,
                content=content,
                error=error,
                started_at=now if status is RequestStatus.RUNNING else current.started_at,
                finished_at=now if status.terminal else None,
            )
            if status.terminal:
                self._running_requests.pop(request_id, None)
            self._record_event_locked(request_id, status, error)
            self._state_condition.notify_all()
            return True

    def _invoke_request(self, request: BackgroundRequest) -> None:
        with self._state_condition:
            current = self._requests[request.request_id]
            if current.status.terminal:
                return
            self._running_requests[request.request_id] = request
            now = time.time()
            self._requests[request.request_id] = replace(
                current, status=RequestStatus.RUNNING, started_at=now
            )
            self._record_event_locked(request.request_id, RequestStatus.RUNNING)
            self._state_condition.notify_all()
        try:
            accepted = self._task_map[request.service_name].invoke_async(
                request.request_id, request.content
            )
            if not accepted:
                self._transition(
                    request.request_id, RequestStatus.FAILED, error="Task rejected request"
                )
        except Exception as exc:
            logger.exception(
                "Task %s failed to accept request %s",
                request.service_name,
                request.request_id,
            )
            self._transition(request.request_id, RequestStatus.FAILED, error=str(exc))

    def _refresh_running(self) -> None:
        with self._state_condition:
            running = list(self._running_requests.values())
        for request in running:
            try:
                response = self._task_map[request.service_name].peek_response(
                    request.request_id
                )
            except (AttributeError, NotImplementedError):
                continue
            except Exception as exc:
                self._transition(request.request_id, RequestStatus.FAILED, error=str(exc))
                continue
            if response.result_type is PeekResponseResultType.ResponseFound:
                self._transition(
                    request.request_id,
                    RequestStatus.SUCCEEDED,
                    content=response.content,
                )
            elif response.result_type is PeekResponseResultType.ErrorUnknown:
                self._transition(
                    request.request_id,
                    RequestStatus.FAILED,
                    error=response.error or "Task execution failed",
                )
            elif (
                response.result_type
                is PeekResponseResultType.ResponseReceivedButFlushed
            ):
                self._transition(
                    request.request_id,
                    RequestStatus.FAILED,
                    error="Task result was flushed before collection",
                )

    def _cancel_nonterminal(self, reason: str) -> None:
        with self._state_condition:
            request_ids = [
                request_id
                for request_id, result in self._requests.items()
                if not result.status.terminal
            ]
        for request_id in request_ids:
            with self._state_condition:
                request = self._running_requests.get(request_id)
            if request is not None:
                cancel = getattr(
                    self._task_map[request.service_name], "cancel_request", None
                )
                if callable(cancel):
                    try:
                        cancel(request_id)
                    except Exception:
                        logger.exception("Failed to cancel request %s", request_id)
            self._transition(request_id, RequestStatus.CANCELLED, error=reason)

    def _bg_thread_loop(self) -> None:
        waiting: deque[TaskPilot.BackgroundRequest] = deque()
        while not self._stop_event.is_set():
            self._wake_event.clear()
            while True:
                try:
                    request = self._task_queue.get_nowait()
                except queue.Empty:
                    break
                with self._state_condition:
                    result = self._requests.get(request.request_id)
                if result is None or result.status.terminal:
                    continue
                if request.associated_move_id < 0:
                    self._invoke_request(request)
                else:
                    self._transition(request.request_id, RequestStatus.WAITING_FOR_MOVE)
                    waiting.append(request)

            if waiting:
                try:
                    robot_state = self._robot_task.query_state()
                except Exception:
                    logger.exception("Failed to query robot state")
                    robot_state = None
                if robot_state is not None:
                    remaining = deque()
                    while waiting:
                        request = waiting.popleft()
                        with self._state_condition:
                            result = self._requests.get(request.request_id)
                        if result is None or result.status.terminal:
                            continue
                        if request.associated_move_id <= robot_state.latest_finished_id:
                            self._invoke_request(request)
                        else:
                            remaining.append(request)
                    waiting = remaining

            self._refresh_running()
            self._wake_event.wait(self._poll_interval)
