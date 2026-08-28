"""Robot and peripheral task orchestration."""

import concurrent.futures
import copy
import heapq
import json
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple

from orchestrion.health import DeviceHealth
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
        priority: int = 0

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
        self._lifecycle_lock = threading.RLock()

        self._task_queue: queue.PriorityQueue[
            Tuple[int, int, TaskPilot.BackgroundRequest]
        ] = queue.PriorityQueue()
        self._next_request_id = 0
        self._state_condition = threading.Condition()
        self._requests: Dict[int, RequestResult] = {}
        self._idempotency_requests: Dict[Tuple[str, str], int] = {}
        self._request_waiters: Dict[int, int] = {}
        self._timeline: deque[TimelineEvent] = deque(maxlen=timeline_capacity)
        self._running_requests: Dict[int, TaskPilot.BackgroundRequest] = {}
        self._health_revision = 0
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._initialized = False

    @property
    def task_map(self) -> Dict[str, GenericTask]:
        """Return a shallow copy of the fixed service registry."""
        return dict(self._task_map)

    @property
    def service_names(self) -> Tuple[str, ...]:
        """Return peripheral service names in registration order."""
        return tuple(self._task_map)

    @property
    def is_running(self) -> bool:
        """Whether the pilot and its scheduler thread are running."""
        thread = self._bg_thread
        return self._initialized and thread is not None and thread.is_alive()

    @property
    def health_revision(self) -> int:
        """Monotonic version incremented when observable runtime state changes."""
        with self._state_condition:
            return self._health_revision

    @property
    def robot_task(self) -> ReducedRobotTaskInterface:
        return self._robot_task

    def __enter__(self) -> "TaskPilot":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def _on_component_change(self) -> None:
        with self._state_condition:
            self._mark_health_changed_locked()
        self._wake_event.set()

    def _mark_health_changed_locked(self) -> None:
        self._health_revision += 1
        self._state_condition.notify_all()

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
        """Initialize the pilot once, serializing concurrent lifecycle calls."""
        with self._lifecycle_lock:
            self._initialize_locked()

    def _initialize_locked(self) -> None:
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
                    set_completion_callback(self._on_component_change)
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
                set_callback(self._on_component_change)
            self._bg_thread = threading.Thread(
                target=self._bg_thread_loop, name="orchestrion-task-pilot"
            )
            self._bg_thread.start()
            with self._state_condition:
                self._initialized = True
                self._mark_health_changed_locked()
        except Exception:
            self._rollback_initialization(initialized_tasks)
            self._bg_thread = None
            raise

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the pilot, serializing against initialization and other stops."""
        with self._lifecycle_lock:
            self._stop_locked(timeout)

    def _stop_locked(self, timeout: float) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(timeout)
            or timeout < 0
        ):
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
                    with self._state_condition:
                        self._initialized = False
                        self._mark_health_changed_locked()
        if thread_timed_out:
            raise TimeoutError("TaskPilot background thread did not stop in time")
        if task_stop_error is not None:
            raise RuntimeError("A peripheral task failed to stop") from task_stop_error

    def call_srv_async(
        self,
        srv_name: str,
        content: Optional[Dict] = None,
        sync_option: Optional[MoveSyncOption] = None,
        priority: int = 0,
        idempotency_key: Optional[str] = None,
    ) -> int:
        if srv_name not in self._task_map:
            raise KeyError("Unknown service: {}".format(srv_name))
        if not self._initialized:
            raise RuntimeError("TaskPilot must be initialized before accepting requests")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError("priority must be an integer")
        if idempotency_key is not None and not isinstance(idempotency_key, str):
            raise TypeError("idempotency_key must be a string or None")
        if idempotency_key == "":
            raise ValueError("idempotency_key must not be empty")
        deduplication_key = (
            None if idempotency_key is None else (srv_name, idempotency_key)
        )
        if deduplication_key is not None:
            with self._state_condition:
                existing_id = self._idempotency_requests.get(deduplication_key)
                if existing_id is not None:
                    return existing_id
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
            if deduplication_key is not None:
                existing_id = self._idempotency_requests.get(deduplication_key)
                if existing_id is not None:
                    return existing_id
            request_id = self._next_request_id
            self._next_request_id += 1
            now = time.time()
            self._requests[request_id] = RequestResult(
                request_id=request_id,
                service_name=srv_name,
                status=RequestStatus.QUEUED,
                associated_move_id=associated_move_id,
                priority=priority,
                idempotency_key=idempotency_key,
                created_at=now,
            )
            if deduplication_key is not None:
                self._idempotency_requests[deduplication_key] = request_id
            self._record_event_locked(request_id, RequestStatus.QUEUED)
            request = self.BackgroundRequest(
                service_name=srv_name,
                request_id=request_id,
                content=request_content,
                associated_move_id=associated_move_id,
                priority=priority,
            )
            self._task_queue.put((-priority, request_id, request))
            self._mark_health_changed_locked()
        self._wake_event.set()
        return request_id

    def query_request(self, request_id: int) -> RequestResult:
        with self._state_condition:
            try:
                return self._requests[request_id]
            except KeyError:
                raise KeyError("Unknown request: {}".format(request_id)) from None

    def query_task_status(self, service_name: str) -> Optional[Dict]:
        try:
            task = self._task_map[service_name]
        except KeyError:
            raise KeyError("Unknown service: {}".format(service_name)) from None
        peek_status = getattr(task, "peek_status", None)
        if not callable(peek_status):
            return None
        status = copy.deepcopy(peek_status())
        if status is None:
            return None
        if not isinstance(status, dict):
            raise TypeError("Task status must be a dictionary or None")
        health = status.setdefault("health", DeviceHealth.ONLINE.value)
        try:
            status["health"] = DeviceHealth(health).value
        except (TypeError, ValueError):
            raise ValueError("Task status contains an invalid health state") from None
        status.setdefault(
            "available", status["health"] != DeviceHealth.OFFLINE.value
        )
        status.setdefault("observed_at", time.time())
        observed_at = status["observed_at"]
        if (
            isinstance(observed_at, bool)
            or not isinstance(observed_at, Real)
            or not math.isfinite(observed_at)
        ):
            raise ValueError("Task status observed_at must be finite")
        return status

    def describe_services(self) -> Dict[str, Dict]:
        """Return static service capabilities without querying device state."""
        descriptions = {}
        for service_name, task in self._task_map.items():
            description = copy.deepcopy(task.describe())
            if not isinstance(description, dict):
                raise TypeError("Task description must be a dictionary")
            description["service_name"] = service_name
            try:
                json.dumps(description, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Task description for {!r} must be JSON-compatible".format(
                        service_name
                    )
                ) from exc
            descriptions[service_name] = description
        return descriptions

    def query_all_task_statuses(self) -> Dict[str, Optional[Dict]]:
        """Return a best-effort snapshot without one device hiding the others."""
        statuses: Dict[str, Optional[Dict]] = {}
        for service_name in self._task_map:
            try:
                statuses[service_name] = self.query_task_status(service_name)
            except Exception as exc:
                statuses[service_name] = {
                    "available": False,
                    "health": DeviceHealth.OFFLINE.value,
                    "observed_at": time.time(),
                    "error": str(exc),
                }
        return statuses

    def query_health(self, stale_after: Optional[float] = None) -> Dict[str, Any]:
        """Return a JSON-compatible health snapshot for monitoring integrations."""
        if stale_after is not None and (
            isinstance(stale_after, bool)
            or not isinstance(stale_after, Real)
            or not math.isfinite(stale_after)
            or stale_after <= 0
        ):
            raise ValueError("stale_after must be positive and finite, or None")
        with self._state_condition:
            request_counts = {status.value: 0 for status in RequestStatus}
            for result in self._requests.values():
                request_counts[result.status.value] += 1
            initialized = self._initialized
            scheduler_alive = (
                self._bg_thread is not None and self._bg_thread.is_alive()
            )
            revision = self._health_revision

        try:
            robot_state = self._robot_task.query_state()
            if robot_state is None:
                robot = {"available": False, "error": "state unavailable"}
            else:
                robot = {
                    "available": True,
                    "joint_positions": list(robot_state.jps),
                    "latest_move_id": robot_state.latest_sent_id,
                    "latest_finished_move_id": robot_state.latest_finished_id,
                }
        except Exception as exc:
            robot = {"available": False, "error": str(exc)}

        peripherals = self.query_all_task_statuses()
        now = time.time()
        if stale_after is not None:
            for status in peripherals.values():
                if status is None:
                    continue
                observed_at = status.get("observed_at")
                stale = (
                    isinstance(observed_at, Real)
                    and not isinstance(observed_at, bool)
                    and math.isfinite(observed_at)
                    and now - observed_at > stale_after
                )
                status["stale"] = stale
                if stale and status.get("health") == DeviceHealth.ONLINE.value:
                    status["health"] = DeviceHealth.DEGRADED.value
        peripherals_available = all(
            status is None
            or (
                status.get("available", True)
                and status.get("health") == DeviceHealth.ONLINE.value
            )
            for status in peripherals.values()
        )
        return {
            "healthy": (
                initialized
                and scheduler_alive
                and robot["available"]
                and peripherals_available
            ),
            "revision": revision,
            "generated_at": now,
            "pilot": {
                "initialized": initialized,
                "scheduler_alive": scheduler_alive,
                "services": list(self.service_names),
            },
            "robot": robot,
            "requests": {
                "total": sum(request_counts.values()),
                **request_counts,
            },
            "peripherals": peripherals,
        }

    def wait_health_change(
        self, after_revision: int, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Wait for observable state newer than ``after_revision``.

        Returns a fresh health snapshot, or ``None`` when the timeout expires.
        """
        if isinstance(after_revision, bool) or not isinstance(after_revision, int):
            raise TypeError("after_revision must be an integer")
        if after_revision < 0:
            raise ValueError("after_revision must be non-negative")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be non-negative and finite, or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state_condition:
            if after_revision > self._health_revision:
                raise ValueError("after_revision is newer than the current revision")
            while self._health_revision <= after_revision:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._state_condition.wait(remaining)
        return self.query_health()

    def wait_request(self, request_id: int, timeout: Optional[float] = None) -> RequestResult:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(timeout)
            or timeout < 0
        ):
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

    def wait_requests(
        self, request_ids: List[int], timeout: Optional[float] = None
    ) -> List[RequestResult]:
        """Wait for requests in order using one shared timeout budget."""
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be non-negative and finite, or None")
        ids = list(request_ids)
        if any(
            isinstance(request_id, bool) or not isinstance(request_id, int)
            for request_id in ids
        ):
            raise TypeError("request_ids must contain integers")
        deadline = None if timeout is None else time.monotonic() + timeout
        results = []
        for request_id in ids:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            results.append(self.wait_request(request_id, timeout=remaining))
        return results

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

    def cancel_all_requests(self, service_name: Optional[str] = None) -> List[int]:
        """Best-effort cancel all nonterminal requests, optionally for one service."""
        if service_name is not None and service_name not in self._task_map:
            raise KeyError("Unknown service: {}".format(service_name))
        with self._state_condition:
            request_ids = [
                request_id
                for request_id, result in self._requests.items()
                if not result.status.terminal
                and (service_name is None or result.service_name == service_name)
            ]
        cancelled = []
        for request_id in request_ids:
            try:
                if self.cancel_request(request_id):
                    cancelled.append(request_id)
            except KeyError:
                continue
        return cancelled

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
                result = self._requests.pop(request_id)
                if result.idempotency_key is not None:
                    self._idempotency_requests.pop(
                        (result.service_name, result.idempotency_key), None
                    )
            if pruned:
                self._mark_health_changed_locked()

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
            self._mark_health_changed_locked()
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
            self._mark_health_changed_locked()
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
        waiting: List[Tuple[int, int, TaskPilot.BackgroundRequest]] = []
        while not self._stop_event.is_set():
            self._wake_event.clear()
            while True:
                try:
                    _, _, request = self._task_queue.get_nowait()
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
                    heapq.heappush(
                        waiting, (-request.priority, request.request_id, request)
                    )

            if waiting:
                try:
                    robot_state = self._robot_task.query_state()
                except Exception:
                    logger.exception("Failed to query robot state")
                    robot_state = None
                if robot_state is not None:
                    remaining = []
                    while waiting:
                        priority_entry = heapq.heappop(waiting)
                        request = priority_entry[2]
                        with self._state_condition:
                            result = self._requests.get(request.request_id)
                        if result is None or result.status.terminal:
                            continue
                        if request.associated_move_id <= robot_state.latest_finished_id:
                            self._invoke_request(request)
                        else:
                            heapq.heappush(remaining, priority_entry)
                    waiting = remaining

            self._refresh_running()
            self._wake_event.wait(self._poll_interval)
