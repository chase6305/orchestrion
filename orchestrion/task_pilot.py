"""Robot and peripheral task orchestration."""

import concurrent.futures
import copy
import heapq
import json
import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Container, Dict, List, Optional, Set, Tuple, Union

from orchestrion.health import DeviceHealth
from orchestrion.move_sync_option import MoveSyncOption
from orchestrion.request import RequestResult, RequestStatus, TimelineEvent
from orchestrion.tasks.generic_task import GenericTask
from orchestrion.tasks.reduced_robot_task_interface import ReducedRobotTaskInterface
from orchestrion.utils.errors import exception_message
from orchestrion.utils.logger import log_error, logger
from orchestrion.utils.numeric import is_finite_real
from orchestrion.utils.types import PeekResponseResult, PeekResponseResultType

# (-priority, request_id, request): higher priorities first, FIFO for ties.
_QueueEntry = Tuple[int, int, "TaskPilot.BackgroundRequest"]


def _validate_wait_timeout(timeout: Optional[float]) -> None:
    """Apply the same timeout contract to request and health waits."""
    if timeout is not None and (not is_finite_real(timeout) or timeout < 0):
        raise ValueError("timeout must be non-negative and finite, or None")


class TaskPilot:
    """Coordinate robot movement and optionally synchronized peripheral requests."""

    @dataclass(frozen=True)
    class BackgroundRequest:
        service_name: str
        request_id: int
        content: Optional[Dict] = None
        associated_move_id: int = -1
        associated_submodule: Optional[str] = None
        priority: int = 0

    def __init__(
        self,
        robot_task: ReducedRobotTaskInterface,
        task_map: Optional[Dict[str, GenericTask]] = None,
        executor_owned: Optional[concurrent.futures.ThreadPoolExecutor] = None,
        poll_interval: float = 0.05,
        timeline_capacity: int = 1000,
    ):
        if not is_finite_real(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be positive and finite")
        if poll_interval > threading.TIMEOUT_MAX:
            raise ValueError("poll_interval must not exceed threading.TIMEOUT_MAX")
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
        self._service_names_by_task_id: Dict[int, Set[str]] = {}
        self._lifecycle_tasks = []
        for name, task in self._task_map.items():
            if id(task) not in self._service_names_by_task_id:
                self._lifecycle_tasks.append((name, task))
            self._service_names_by_task_id.setdefault(id(task), set()).add(name)
        self._executor = executor_owned
        self._executor_shutdown = False
        self._poll_interval = poll_interval
        self._lifecycle_lock = threading.RLock()

        self._task_queue: queue.PriorityQueue[_QueueEntry] = queue.PriorityQueue()
        self._next_request_id = 0
        self._state_condition = threading.Condition()
        self._requests: Dict[int, RequestResult] = {}
        self._request_counts = {status.value: 0 for status in RequestStatus}
        self._idempotency_requests: Dict[Tuple[str, str], int] = {}
        self._request_waiters: Dict[int, int] = {}
        self._timeline: deque[TimelineEvent] = deque(maxlen=timeline_capacity)
        self._running_requests: Dict[int, TaskPilot.BackgroundRequest] = {}
        self._cancelling_requests: Dict[
            int, Optional[Tuple[RequestStatus, Any, Optional[str]]]
        ] = {}
        self._health_revision = 0
        self._move_state_revision = 0
        self._dirty_services: Set[str] = set()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._scheduler_ready = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._initialized = False
        self._lifecycle_epoch = 0

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
        """Version advanced by pilot transitions and adapter notifications."""
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
            self._move_state_revision += 1
            self._mark_health_changed_locked()
        self._wake_event.set()

    def _on_task_completion(self, service_name: str) -> None:
        with self._state_condition:
            self._dirty_services.update(
                self._service_names_by_task_id[id(self._task_map[service_name])]
            )
            self._mark_health_changed_locked()
        self._wake_event.set()

    def _mark_health_changed_locked(self) -> None:
        self._health_revision += 1
        self._state_condition.notify_all()

    def _rollback_initialization(self, initialized_tasks: List[GenericTask]) -> None:
        for task in reversed(initialized_tasks):
            try:
                task.stop()
            except Exception as exc:
                log_error(
                    "Failed to roll back peripheral task: %s", exception_message(exc)
                )
        for _, task in self._lifecycle_tasks:
            try:
                set_completion_callback = getattr(task, "set_completion_callback", None)
                if callable(set_completion_callback):
                    set_completion_callback(None)
            except Exception as exc:
                log_error(
                    "Failed to clear task completion callback: %s",
                    exception_message(exc),
                )
        try:
            set_state_callback = getattr(
                self._robot_task, "set_state_change_callback", None
            )
            if callable(set_state_callback):
                set_state_callback(None)
        except Exception as exc:
            log_error(
                "Failed to clear robot state callback: %s", exception_message(exc)
            )
        try:
            self._robot_task.stop()
        except Exception as exc:
            log_error("Failed to roll back robot task: %s", exception_message(exc))

    def initialize(self) -> None:
        """Initialize the pilot once, serializing concurrent lifecycle calls."""
        with self._lifecycle_lock:
            self._initialize_locked()

    def _initialize_locked(self) -> None:
        if self._initialized:
            logger.warning("TaskPilot is already initialized.")
            return
        if self._bg_thread is not None and self._bg_thread.is_alive():
            raise RuntimeError("TaskPilot background thread is still stopping")
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
            except Exception as exc:
                log_error(
                    "Failed to roll back robot task initialization: %s",
                    exception_message(exc),
                )
            raise
        initialized_tasks = []
        try:
            for service_name, task in self._lifecycle_tasks:
                set_completion_callback = getattr(task, "set_completion_callback", None)
                if callable(set_completion_callback):
                    set_completion_callback(
                        partial(self._on_task_completion, service_name)
                    )
                initialized_tasks.append(task)
                task.initialize(executor=self._executor)
        except Exception:
            self._rollback_initialization(initialized_tasks)
            raise

        self._scheduler_ready.clear()
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
            with self._state_condition:
                self._initialized = False
                self._lifecycle_epoch += 1
                self._stop_event.set()
            self._cancel_nonterminal("TaskPilot initialization failed")
            self._rollback_initialization(initialized_tasks)
            self._bg_thread = None
            raise
        finally:
            self._scheduler_ready.set()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the pilot, serializing against initialization and other stops."""
        if threading.current_thread() is self._bg_thread:
            self._stop_locked(timeout)
            return
        with self._lifecycle_lock:
            self._stop_locked(timeout)

    def _stop_locked(
        self, timeout: float, *, reason: str = "TaskPilot stopped"
    ) -> None:
        if not is_finite_real(timeout) or timeout < 0:
            raise ValueError("timeout must be non-negative and finite")
        if timeout > threading.TIMEOUT_MAX:
            raise ValueError("timeout must not exceed threading.TIMEOUT_MAX")
        timeout = float(timeout)
        # Elect one cleanup owner before calling device code. A scheduler callback
        # can request stop while an external caller is already joining that thread.
        with self._state_condition:
            was_initialized = self._initialized
            if was_initialized:
                self._initialized = False
                self._lifecycle_epoch += 1
                self._stop_event.set()
                self._mark_health_changed_locked()
        if not was_initialized:
            thread = self._bg_thread
            if (
                thread is not None
                and thread is not threading.current_thread()
                and thread.is_alive()
            ):
                thread.join(timeout=timeout)
                if thread.is_alive():
                    raise TimeoutError(
                        "TaskPilot background thread did not stop in time"
                    )
            logger.warning("TaskPilot is not running or already stopped.")
            return
        self._wake_event.set()
        thread_timed_out = False
        if (
            self._bg_thread is not None
            and self._bg_thread is not threading.current_thread()
        ):
            self._bg_thread.join(timeout=timeout)
            thread_timed_out = self._bg_thread.is_alive()

        self._cancel_nonterminal(reason)
        while True:
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                break
        task_stop_error: Optional[Exception] = None
        try:
            for _, task in self._lifecycle_tasks:
                try:
                    task.stop()
                except Exception as exc:
                    log_error(
                        "Failed to stop peripheral task: %s", exception_message(exc)
                    )
                    if task_stop_error is None:
                        task_stop_error = exc
                finally:
                    try:
                        set_completion_callback = getattr(
                            task, "set_completion_callback", None
                        )
                        if callable(set_completion_callback):
                            set_completion_callback(None)
                    except Exception as exc:
                        log_error(
                            "Failed to clear task completion callback: %s",
                            exception_message(exc),
                        )
        finally:
            try:
                if self._executor is not None:
                    # Running Python callbacks cannot be forcefully interrupted.
                    # Shut down admission immediately and cancel queued calls;
                    # running callbacks finish on their existing worker threads.
                    self._executor.shutdown(wait=False, cancel_futures=True)
                    self._executor_shutdown = True
            finally:
                try:
                    set_callback = getattr(
                        self._robot_task, "set_state_change_callback", None
                    )
                    if callable(set_callback):
                        set_callback(None)
                except Exception as exc:
                    log_error(
                        "Failed to clear robot state callback: %s",
                        exception_message(exc),
                    )
                try:
                    self._robot_task.stop()
                finally:
                    with self._state_condition:
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
        """Queue a service request; log and return -1 when submission is rejected.

        Failed preparation does not consume a request ID or idempotency key.
        Accepted requests expose execution failures through wait_request().
        """
        try:
            result = self._submit_service_request(
                srv_name, content, sync_option, priority, idempotency_key
            )
        except Exception as exc:
            result = exception_message(exc)
        if isinstance(result, str):
            log_error("Service request was not submitted: %s", result)
            return -1
        return result

    def _submit_service_request(
        self,
        srv_name: str,
        content: Optional[Dict],
        sync_option: Optional[MoveSyncOption],
        priority: int,
        idempotency_key: Optional[str],
    ) -> Union[int, str]:
        """Prepare and atomically admit a request behind the nonthrowing boundary."""
        if not isinstance(srv_name, str):
            return "srv_name must be a string"
        srv_name = str.__str__(srv_name)
        if srv_name not in self._task_map:
            return "Unknown service: {}".format(srv_name)
        if not self._initialized:
            return "TaskPilot must be initialized before accepting requests"
        with self._state_condition:
            submission_epoch = self._lifecycle_epoch
        if isinstance(priority, bool) or not isinstance(priority, int):
            return "priority must be an integer"
        priority = int.__int__(priority)
        if idempotency_key is not None and not isinstance(idempotency_key, str):
            return "idempotency_key must be a string or None"
        if idempotency_key is not None:
            idempotency_key = str.__str__(idempotency_key)
        if idempotency_key == "":
            return "idempotency_key must not be empty"
        deduplication_key = (
            None if idempotency_key is None else (srv_name, idempotency_key)
        )
        if deduplication_key is not None:
            with self._state_condition:
                existing_id = self._idempotency_requests.get(deduplication_key)
                if existing_id is not None:
                    return existing_id
        if sync_option is None:
            sync_option = MoveSyncOption.sync_w_latest_move()
        elif not isinstance(sync_option, MoveSyncOption):
            return "sync_option must be a MoveSyncOption or None"
        else:
            # Options are intentionally mutable. Validate a plain-value snapshot
            # before SDK callbacks can mutate them or invalid fields reach the
            # background scheduler's hash keys and movement comparisons.
            snapshot = MoveSyncOption._snapshot(sync_option)
            if isinstance(snapshot, str):
                return snapshot
            sync_option = snapshot

        associated_move_id = -1
        associated_submodule = sync_option.submodule_name
        if sync_option.need_sync:
            if associated_submodule is not None:
                query_submodule_state = getattr(
                    self._robot_task, "query_submodule_state", None
                )
                if not callable(query_submodule_state):
                    return "Robot task does not support submodule synchronization"
                submodule_state = query_submodule_state(associated_submodule)
                associated_move_id = (
                    sync_option.associated_move_id
                    if sync_option.associated_move_id >= 0
                    else submodule_state.latest_sent_id
                )
            elif sync_option.associated_move_id >= 0:
                associated_move_id = sync_option.associated_move_id
            else:
                robot_state = self._robot_task.query_state()
                if robot_state is None:
                    return "Robot state is unavailable"
                associated_move_id = robot_state.latest_sent_id
            if isinstance(associated_move_id, bool) or not isinstance(
                associated_move_id, int
            ):
                return "associated_move_id must be an integer"
            associated_move_id = int.__int__(associated_move_id)
            if associated_move_id < -1:
                return "associated_move_id must be -1 or non-negative"

        request_content = copy.deepcopy(content)
        with self._state_condition:
            if not self._initialized or submission_epoch != self._lifecycle_epoch:
                return "TaskPilot stopped while the request was being prepared"
            if deduplication_key is not None:
                existing_id = self._idempotency_requests.get(deduplication_key)
                if existing_id is not None:
                    return existing_id
            request_id = self._next_request_id
            self._next_request_id += 1
            now = time.time()
            self._store_request_locked(
                RequestResult(
                    request_id=request_id,
                    service_name=srv_name,
                    status=RequestStatus.QUEUED,
                    associated_move_id=associated_move_id,
                    associated_submodule=associated_submodule,
                    priority=priority,
                    idempotency_key=idempotency_key,
                    created_at=now,
                )
            )
            if deduplication_key is not None:
                self._idempotency_requests[deduplication_key] = request_id
            self._record_event_locked(request_id, RequestStatus.QUEUED)
            request = self.BackgroundRequest(
                service_name=srv_name,
                request_id=request_id,
                content=request_content,
                associated_move_id=associated_move_id,
                associated_submodule=associated_submodule,
                priority=priority,
            )
            self._task_queue.put((-priority, request_id, request))
            self._mark_health_changed_locked()
        self._wake_event.set()
        return request_id

    @staticmethod
    def _validate_request_id(request_id: int) -> None:
        if isinstance(request_id, bool) or not isinstance(request_id, int):
            raise TypeError("request_id must be an integer")
        if request_id < 0:
            raise ValueError("request_id must be non-negative")

    def query_request(self, request_id: int) -> RequestResult:
        self._validate_request_id(request_id)
        with self._state_condition:
            try:
                result = self._requests[request_id]
            except KeyError:
                raise KeyError("Unknown request: {}".format(request_id)) from None
        return copy.deepcopy(result)

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
        if not isinstance(status["available"], bool):
            raise ValueError("Task status available must be a boolean")
        status.setdefault("observed_at", time.time())
        observed_at = status["observed_at"]
        if not is_finite_real(observed_at):
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
                    "error": exception_message(exc),
                }
        return statuses

    def query_health(self, stale_after: Optional[float] = None) -> Dict[str, Any]:
        """Return a JSON-compatible health snapshot for monitoring integrations."""
        if stale_after is not None and (
            not is_finite_real(stale_after) or stale_after <= 0
        ):
            raise ValueError("stale_after must be positive and finite, or None")
        with self._state_condition:
            request_counts = self._request_counts.copy()
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
                json.dumps(robot, allow_nan=False)
        except Exception as exc:
            robot = {"available": False, "error": exception_message(exc)}

        peripherals = self.query_all_task_statuses()
        for service_name, status in peripherals.items():
            try:
                json.dumps(status, allow_nan=False)
            except (TypeError, ValueError) as exc:
                peripherals[service_name] = {
                    "available": False,
                    "health": DeviceHealth.OFFLINE.value,
                    "observed_at": time.time(),
                    "error": "Task status is not JSON-compatible: {}".format(
                        exception_message(exc)
                    ),
                }
        now = time.time()
        if stale_after is not None:
            for status in peripherals.values():
                if status is None:
                    continue
                observed_at = status.get("observed_at")
                stale = (
                    is_finite_real(observed_at)
                    and now - observed_at > stale_after
                )
                status["stale"] = stale
                if stale and status.get("health") == DeviceHealth.ONLINE.value:
                    status["health"] = DeviceHealth.DEGRADED.value
        peripherals_healthy = all(
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
                and peripherals_healthy
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
        """Wait for a pilot transition or adapter notification after a revision.

        Returns a fresh health snapshot, or ``None`` when the timeout expires.
        Passive device changes need query_health(); sample aging is checked with
        query_health(stale_after=...). Neither change automatically notifies.
        Blocking waits are not allowed from this pilot's scheduler callbacks.
        """
        if isinstance(after_revision, bool) or not isinstance(after_revision, int):
            raise TypeError("after_revision must be an integer")
        if after_revision < 0:
            raise ValueError("after_revision must be non-negative")
        _validate_wait_timeout(timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state_condition:
            if after_revision > self._health_revision:
                raise ValueError("after_revision is newer than the current revision")
            while self._health_revision <= after_revision:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._check_scheduler_wait()
                # Bound each native wait while preserving the overall deadline.
                if remaining is not None:
                    remaining = min(remaining, threading.TIMEOUT_MAX)
                self._state_condition.wait(remaining)
        return self.query_health()

    def _check_scheduler_wait(self) -> None:
        if threading.current_thread() is self._bg_thread:
            raise RuntimeError(
                "TaskPilot scheduler callbacks cannot block waiting on their own "
                "pilot; submit follow-up work asynchronously"
            )

    def wait_request(
        self, request_id: int, timeout: Optional[float] = None
    ) -> RequestResult:
        """Wait for a result; scheduler callbacks may only read completed results."""
        self._validate_request_id(request_id)
        _validate_wait_timeout(timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state_condition:
            if request_id not in self._requests:
                raise KeyError("Unknown request: {}".format(request_id))
            self._request_waiters[request_id] = (
                self._request_waiters.get(request_id, 0) + 1
            )
        try:
            with self._state_condition:
                while not self._requests[request_id].status.terminal:
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError(
                            "Request {} did not finish in time".format(request_id)
                        )
                    self._check_scheduler_wait()
                    if remaining is not None:
                        remaining = min(remaining, threading.TIMEOUT_MAX)
                    self._state_condition.wait(remaining)
                result = self._requests[request_id]
            # Keep the waiter registered while copying outside the scheduler lock.
            return copy.deepcopy(result)
        finally:
            with self._state_condition:
                self._release_request_waiter_locked(request_id)

    def wait_requests(
        self, request_ids: List[int], timeout: Optional[float] = None
    ) -> List[RequestResult]:
        """Wait in order with a shared deadline, retaining every group's result."""
        _validate_wait_timeout(timeout)
        ids = list(request_ids)
        if any(
            isinstance(request_id, bool) or not isinstance(request_id, int)
            for request_id in ids
        ):
            raise TypeError("request_ids must contain integers")
        deadline = None if timeout is None else time.monotonic() + timeout
        unique_ids = set(ids)
        with self._state_condition:
            for request_id in ids:
                self._validate_request_id(request_id)
                if request_id not in self._requests:
                    raise KeyError("Unknown request: {}".format(request_id))
            for request_id in unique_ids:
                self._request_waiters[request_id] = (
                    self._request_waiters.get(request_id, 0) + 1
                )
        try:
            results = []
            for request_id in ids:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                results.append(self.wait_request(request_id, timeout=remaining))
            return results
        finally:
            with self._state_condition:
                for request_id in unique_ids:
                    self._release_request_waiter_locked(request_id)

    def _release_request_waiter_locked(self, request_id: int) -> None:
        waiter_count = self._request_waiters[request_id] - 1
        if waiter_count:
            self._request_waiters[request_id] = waiter_count
        else:
            self._request_waiters.pop(request_id)

    def cancel_request(self, request_id: int) -> bool:
        self._validate_request_id(request_id)
        with self._state_condition:
            result = self._requests.get(request_id)
            if result is None:
                raise KeyError("Unknown request: {}".format(request_id))
            if result.status.terminal or request_id in self._cancelling_requests:
                return False
            running = request_id in self._running_requests
            if not running:
                self._transition(
                    request_id, RequestStatus.CANCELLED, error="Cancelled by user"
                )
                self._wake_event.set()
                return True
            cancel = getattr(
                self._task_map[result.service_name], "cancel_request", None
            )
            if not callable(cancel):
                return False
            self._cancelling_requests[request_id] = None

        accepted = False
        cancelled = False
        try:
            # Backend cancellation may wait for a worker that notifies the pilot.
            # Do not hold the shared state lock across arbitrary device code.
            accepted = bool(cancel(request_id))
        finally:
            with self._state_condition:
                deferred = self._cancelling_requests.pop(request_id)
                if accepted:
                    cancelled = self._transition(
                        request_id, RequestStatus.CANCELLED, error="Cancelled by user"
                    )
                elif deferred is not None:
                    status, content, error = deferred
                    self._transition(request_id, status, content=content, error=error)
            self._wake_event.set()
        return cancelled

    def cancel_all_requests(self, service_name: Optional[str] = None) -> List[int]:
        """Attempt every selected cancellation, logging individual backend errors.

        Return only IDs whose cancellation was accepted. Failed attempts retain
        their observed state and do not prevent cancellation of other requests.
        """
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
            except Exception as exc:
                log_error(
                    "Failed to cancel request %s: %s",
                    request_id,
                    exception_message(exc),
                )
        return cancelled

    def timeline(self, request_id: Optional[int] = None) -> List[TimelineEvent]:
        if request_id is not None:
            self._validate_request_id(request_id)
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
                self._request_counts[result.status.value] -= 1
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
        move_id_begin = self._robot_task.move_joint_trajectory_async(
            motion_target=motion_target,
            interval=interval,
            endpoint_index=endpoint_index,
        )
        if move_id_begin < 0:
            return move_id_begin, move_id_begin
        n_segments = 1 if endpoint_index is None else len(endpoint_index)
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

    def _store_request_locked(self, result: RequestResult) -> None:
        previous = self._requests.get(result.request_id)
        if previous is not None:
            self._request_counts[previous.status.value] -= 1
        self._requests[result.request_id] = result
        self._request_counts[result.status.value] += 1

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
            if (
                status.terminal
                and status is not RequestStatus.CANCELLED
                and request_id in self._cancelling_requests
            ):
                if self._cancelling_requests[request_id] is None:
                    self._cancelling_requests[request_id] = (status, content, error)
                return False
            now = time.time()
            self._store_request_locked(
                replace(
                    current,
                    status=status,
                    content=content,
                    error=error,
                    started_at=(
                        now if status is RequestStatus.RUNNING else current.started_at
                    ),
                    finished_at=now if status.terminal else None,
                )
            )
            if status.terminal:
                self._running_requests.pop(request_id, None)
            self._record_event_locked(request_id, status, error)
            self._mark_health_changed_locked()
            return True

    def _invoke_request(self, request: BackgroundRequest) -> None:
        with self._state_condition:
            current = self._requests.get(request.request_id)
            if (
                self._stop_event.is_set()
                or not self._initialized
                or current is None
                or current.status.terminal
            ):
                return
            self._running_requests[request.request_id] = request
            now = time.time()
            self._store_request_locked(
                replace(current, status=RequestStatus.RUNNING, started_at=now)
            )
            self._record_event_locked(request.request_id, RequestStatus.RUNNING)
            self._mark_health_changed_locked()
        try:
            accepted = self._task_map[request.service_name].invoke_async(
                request.request_id, request.content
            )
            if not accepted:
                self._fail_request(request, "Task rejected request")
        except Exception as exc:
            self._fail_request(request, exception_message(exc))

    def _fail_request(self, request: BackgroundRequest, error: str) -> None:
        if self._transition(request.request_id, RequestStatus.FAILED, error=error):
            log_error(
                "Task %s request %s failed: %s",
                request.service_name,
                request.request_id,
                error,
            )

    def _refresh_running(self, service_names: Optional[Set[str]] = None) -> None:
        with self._state_condition:
            running = [
                request for request in self._running_requests.values()
                if service_names is None or request.service_name in service_names
            ]
        for request in running:
            self._refresh_request(request)

    def _refresh_request(self, request: BackgroundRequest) -> None:
        with self._state_condition:
            if (
                request.request_id not in self._running_requests
                or self._cancelling_requests.get(request.request_id) is not None
            ):
                return
        try:
            peek_response = getattr(
                self._task_map[request.service_name], "peek_response", None
            )
            if not callable(peek_response):
                return
            response = peek_response(request.request_id)
            if not isinstance(response, PeekResponseResult) or not isinstance(
                response.result_type, PeekResponseResultType
            ):
                self._fail_request(
                    request, "Task returned an invalid response envelope"
                )
                return
            if isinstance(response.request_id, bool) or not isinstance(
                response.request_id, int
            ):
                self._fail_request(
                    request, "Task response request_id must be an integer"
                )
                return
            if response.request_id not in (-1, request.request_id):
                self._fail_request(
                    request, "Task returned a response for a different request"
                )
                return
            if response.error is not None and not isinstance(response.error, str):
                self._fail_request(
                    request, "Task response error must be a string or None"
                )
                return
            if not isinstance(response.cancelled, bool) or (
                response.cancelled
                and response.result_type is not PeekResponseResultType.ErrorUnknown
            ):
                self._fail_request(
                    request, "Task response has an invalid cancellation flag"
                )
                return
            if response.result_type is PeekResponseResultType.ResponseFound:
                try:
                    content = copy.deepcopy(response.content)
                except Exception as exc:
                    self._fail_request(
                        request,
                        "Task response could not be copied: {}".format(
                            exception_message(exc)
                        ),
                    )
                    return
                self._transition(
                    request.request_id,
                    RequestStatus.SUCCEEDED,
                    content=content,
                )
            elif response.result_type is PeekResponseResultType.ErrorUnknown:
                if response.cancelled:
                    self._transition(
                        request.request_id,
                        RequestStatus.CANCELLED,
                        error=response.error or "Task execution cancelled",
                    )
                else:
                    self._fail_request(
                        request, response.error or "Task execution failed"
                    )
            elif (
                response.result_type
                is PeekResponseResultType.ResponseReceivedButFlushed
            ):
                self._fail_request(request, "Task result was flushed before collection")
            elif response.result_type is PeekResponseResultType.ErrorRequestNotSent:
                self._fail_request(
                    request, "Task has no record of the accepted request"
                )
        except NotImplementedError:
            # Legacy adapters may only implement fire-and-forget invocation.
            return
        except Exception as exc:
            self._fail_request(request, exception_message(exc))

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
                cancellation_in_progress = request_id in self._cancelling_requests
            if request is not None and not cancellation_in_progress:
                try:
                    cancel = getattr(
                        self._task_map[request.service_name], "cancel_request", None
                    )
                    if callable(cancel):
                        cancel(request_id)
                except Exception as exc:
                    log_error(
                        "Failed to cancel request %s: %s",
                        request_id,
                        exception_message(exc),
                    )
            self._transition(request_id, RequestStatus.CANCELLED, error=reason)

    def _admit_queued_requests(
        self,
        waiting_for_motion: List[_QueueEntry],
        ready_to_dispatch: List[_QueueEntry],
    ) -> bool:
        """Admit a bounded batch and report whether it adds motion dependencies."""
        added_motion_dependency = False
        # A fixed batch lets response collection and shutdown keep up with producers.
        for _ in range(self._task_queue.qsize()):
            if self._stop_event.is_set():
                break
            try:
                queue_entry = self._task_queue.get_nowait()
            except queue.Empty:
                break
            request = queue_entry[2]
            with self._state_condition:
                result = self._requests.get(request.request_id)
            if result is None or result.status.terminal:
                continue
            if request.associated_move_id < 0:
                heapq.heappush(ready_to_dispatch, queue_entry)
            else:
                self._transition(request.request_id, RequestStatus.WAITING_FOR_MOVE)
                waiting_for_motion.append(queue_entry)
                added_motion_dependency = True
        return added_motion_dependency

    def _promote_completed_moves(
        self,
        waiting_for_motion: List[_QueueEntry],
        ready_to_dispatch: List[_QueueEntry],
    ) -> List[_QueueEntry]:
        """Resolve shared motion snapshots and retain unfinished dependencies."""
        dependency_counts = Counter(
            entry[2].associated_submodule for entry in waiting_for_motion
        )
        main_state = None
        if None in dependency_counts:
            try:
                main_state = self._robot_task.query_state()
            except Exception as exc:
                # A transient main-state failure leaves its requests waiting.
                log_error("Failed to query robot state: %s", exception_message(exc))

        # Index shared sequences only within this pass. Sets and short batches
        # keep their existing membership lookup without an extra allocation.
        progress_by_submodule: Dict[
            Optional[str], Union[Tuple[int, Container[int]], str]
        ] = {}
        still_waiting = []
        for queue_entry in waiting_for_motion:
            request = queue_entry[2]
            with self._state_condition:
                result = self._requests.get(request.request_id)
            if result is None or result.status.terminal:
                continue
            submodule_name = request.associated_submodule
            if submodule_name is None and main_state is None:
                still_waiting.append(queue_entry)
                continue
            if submodule_name not in progress_by_submodule:
                progress = self._read_motion_progress(
                    submodule_name, main_state, dependency_counts[submodule_name]
                )
                progress_by_submodule[submodule_name] = progress
                if isinstance(progress, str):
                    log_error(
                        "Failed to read motion progress for %s: %s",
                        "main robot" if submodule_name is None else submodule_name,
                        progress,
                    )
            progress = progress_by_submodule[submodule_name]
            if isinstance(progress, str):
                self._transition(
                    request.request_id, RequestStatus.FAILED, error=progress
                )
                continue
            try:
                latest_finished_id, cancelled_moves = progress
                if request.associated_move_id in cancelled_moves:
                    self._transition(
                        request.request_id,
                        RequestStatus.CANCELLED,
                        error=(
                            "Associated move was cancelled" if submodule_name is None
                            else "Associated submodule move was cancelled"
                        ),
                    )
                    continue
                move_finished = request.associated_move_id <= latest_finished_id
            except Exception as exc:
                self._fail_request(request, exception_message(exc))
                continue
            if move_finished:
                heapq.heappush(ready_to_dispatch, queue_entry)
            else:
                still_waiting.append(queue_entry)
        return still_waiting

    def _read_motion_progress(
        self,
        submodule_name: Optional[str],
        main_state: Optional[ReducedRobotTaskInterface.ReducedRobotState],
        dependency_count: int,
    ) -> Union[Tuple[int, Container[int]], str]:
        """Return a progress snapshot or one reusable error for this scheduler pass."""
        try:
            if submodule_name is None:
                cancelled_move_ids = getattr(main_state, "cancelled_move_ids", ())
                latest_finished_id = main_state.latest_finished_id
            else:
                state = self._robot_task.query_submodule_state(submodule_name)
                cancelled_move_ids = state.cancelled_move_ids
                latest_finished_id = state.latest_finished_id
            if isinstance(latest_finished_id, bool) or not isinstance(
                latest_finished_id, int
            ):
                return "Robot latest_finished_id must be an integer"
            latest_finished_id = int.__int__(latest_finished_id)
            if latest_finished_id < -1:
                return "Robot latest_finished_id must be -1 or non-negative"
            cancellation_lookup = (
                set(cancelled_move_ids)
                if dependency_count >= 4
                and isinstance(cancelled_move_ids, (list, tuple))
                else cancelled_move_ids
            )
            return latest_finished_id, cancellation_lookup
        except Exception as exc:
            return exception_message(exc)

    def _dispatch_next_request(self, ready_to_dispatch: List[_QueueEntry]) -> None:
        request = heapq.heappop(ready_to_dispatch)[2]
        self._invoke_request(request)
        # Collect inline results before the next call can evict a bounded response.
        self._refresh_request(request)

    def _bg_thread_loop(self) -> None:
        # Startup must publish admission before the worker can report a failure;
        # otherwise an early worker exit could be overwritten by initialize().
        self._scheduler_ready.wait()
        if not self._initialized:
            return
        try:
            self._run_scheduler()
        except Exception as exc:
            message = exception_message(exc)
            log_error("TaskPilot scheduler failed: %s", message)
            try:
                self._stop_locked(0.0, reason="TaskPilot scheduler failed: " + message)
            except Exception as cleanup_error:
                log_error(
                    "TaskPilot scheduler cleanup failed: %s",
                    exception_message(cleanup_error),
                )

    def _run_scheduler(self) -> None:
        waiting_for_motion: List[_QueueEntry] = []
        ready_to_dispatch: List[_QueueEntry] = []
        next_poll_at = time.monotonic()
        observed_motion_revision = -1
        while not self._stop_event.is_set():
            self._wake_event.clear()
            poll_due = time.monotonic() >= next_poll_at
            added_motion_dependency = self._admit_queued_requests(
                waiting_for_motion, ready_to_dispatch
            )
            with self._state_condition:
                motion_revision = self._move_state_revision
                services_with_updates = self._dirty_services
                self._dirty_services = set()

            if waiting_for_motion and (
                added_motion_dependency
                or poll_due
                or motion_revision != observed_motion_revision
            ):
                observed_motion_revision = motion_revision
                waiting_for_motion = self._promote_completed_moves(
                    waiting_for_motion, ready_to_dispatch
                )

            if poll_due:
                self._refresh_running()
                next_poll_at = time.monotonic() + self._poll_interval
            elif services_with_updates:
                self._refresh_running(services_with_updates)

            # Dispatch one request before checking new priorities and collecting
            # results again. Helper frames release payloads and snapshots on return.
            if ready_to_dispatch:
                self._dispatch_next_request(ready_to_dispatch)
            if not ready_to_dispatch and self._task_queue.empty():
                with self._state_condition:
                    needs_polling = bool(waiting_for_motion or self._running_requests)
                timeout = (
                    max(0.0, next_poll_at - time.monotonic()) if needs_polling else None
                )
                self._wake_event.wait(timeout)
