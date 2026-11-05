"""Background polling adapter for sensor and device telemetry services."""

import copy
import threading
import time
from typing import Callable, Dict, Optional

from orchestrion.health import DeviceHealth
from orchestrion.tasks.function_call_task import InPlaceFunctionCallTask
from orchestrion.utils.errors import exception_message
from orchestrion.utils.numeric import is_finite_real


class PollingTask(InPlaceFunctionCallTask):
    """Poll a device in the background and serve its latest cached sample."""

    task_kind = "telemetry"
    execution_mode = "background_polling"

    def __init__(
        self,
        poll: Callable[[], Optional[Dict]],
        interval: float = 1.0,
        max_result_count: int = -1,
        metadata: Optional[Dict] = None,
    ):
        if not callable(poll):
            raise TypeError("poll must be callable")
        if not is_finite_real(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        if interval > threading.TIMEOUT_MAX:
            raise ValueError("interval must not exceed threading.TIMEOUT_MAX")
        super().__init__(max_result_count=max_result_count, metadata=metadata)
        self._poll = poll
        self._interval = float(interval)
        self._sample_lock = threading.Lock()
        self._latest_sample: Optional[Dict] = None
        self._observed_at: Optional[float] = None
        self._last_error: Optional[str] = None
        self._consecutive_failures = 0
        self._poll_stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_lifecycle_lock = threading.RLock()

    def describe(self) -> Dict:
        description = super().describe()
        description["poll_interval"] = self._interval
        return description

    def initialize(self, **kwargs) -> None:
        with self._poll_lifecycle_lock:
            if self._poll_thread is not None and self._poll_thread.is_alive():
                if self._poll_stop.is_set():
                    raise RuntimeError("Polling task thread is still stopping")
                return
            self._poll_stop.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_loop,
                name="orchestrion-polling-task",
                daemon=True,
            )
            try:
                self._poll_thread.start()
            except Exception:
                self._poll_thread = None
                self._poll_stop.set()
                raise

    def stop(self) -> None:
        if threading.current_thread() is self._poll_thread:
            # An external stop may be joining this worker while holding the
            # lifecycle lock. Worker callbacks must not wait for that lock.
            was_stopped = self._poll_stop.is_set()
            self._poll_stop.set()
        else:
            with self._poll_lifecycle_lock:
                was_stopped = self._poll_stop.is_set()
                self._poll_stop.set()
                thread = self._poll_thread
                if thread is not None and thread is not threading.current_thread():
                    # SDK calls need their own I/O timeout; keep framework
                    # shutdown bounded if a poll function blocks.
                    thread.join(timeout=1.0)
                    if not thread.is_alive():
                        self._poll_thread = None
        if not was_stopped:
            self._notify_completion()

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            previous_sample = None
            try:
                sample = self._poll()
                if sample is not None and not isinstance(sample, dict):
                    raise TypeError("poll callable must return a dictionary or None")
                sample = copy.deepcopy(sample)
                if sample is not None and not isinstance(sample, dict):
                    raise TypeError("copied poll sample must be a dictionary or None")
                with self._sample_lock:
                    previous_sample = self._latest_sample
                    self._latest_sample = sample
                    self._observed_at = time.time()
                    self._last_error = None
                    self._consecutive_failures = 0
            except Exception as exc:
                message = exception_message(exc)
                with self._sample_lock:
                    self._last_error = message
                    self._consecutive_failures += 1
            # Release retired and rejected SDK samples without the cache lock.
            del previous_sample
            sample = None
            self._notify_completion()
            self._poll_stop.wait(self._interval)

    def _call_fn(self, request_id: int, content: Optional[Dict]) -> Optional[Dict]:
        with self._sample_lock:
            if self._observed_at is None:
                raise RuntimeError(
                    self._last_error or "No telemetry sample is available"
                )
            sample = self._latest_sample
            observed_at = self._observed_at
        # Published samples are replaced, never mutated; copying needs no lock.
        return {
            "sample": copy.deepcopy(sample),
            "observed_at": observed_at,
        }

    def peek_status(self) -> Dict:
        status = super().peek_status()
        with self._sample_lock:
            observed_at = self._observed_at
            last_error = self._last_error
            failures = self._consecutive_failures
        thread = self._poll_thread
        initialized = (
            thread is not None
            and thread.is_alive()
            and not self._poll_stop.is_set()
        )
        if not initialized:
            health = DeviceHealth.OFFLINE
        elif observed_at is None:
            health = DeviceHealth.OFFLINE if last_error else DeviceHealth.CONNECTING
        elif failures:
            health = DeviceHealth.DEGRADED
        else:
            health = DeviceHealth.ONLINE
        status.update(
            {
                "health": health.value,
                "available": initialized and observed_at is not None,
                "initialized": initialized,
                "observed_at": observed_at or time.time(),
                "last_error": last_error,
                "consecutive_failures": failures,
            }
        )
        return status
