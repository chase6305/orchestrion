"""Background polling adapter for sensor and device telemetry services."""

import copy
import math
import threading
import time
from numbers import Real
from typing import Callable, Dict, Optional

from orchestrion.health import DeviceHealth
from orchestrion.tasks.function_call_task import InPlaceFunctionCallTask


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
        if (
            isinstance(interval, bool)
            or not isinstance(interval, Real)
            or not math.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError("interval must be positive and finite")
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

    def describe(self) -> Dict:
        description = super().describe()
        description["poll_interval"] = self._interval
        return description

    def initialize(self, **kwargs) -> None:
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="orchestrion-polling-task",
            daemon=True,
        )
        self._poll_thread.start()

    def stop(self) -> None:
        self._poll_stop.set()
        thread = self._poll_thread
        if thread is not None:
            # Device SDK calls must configure their own I/O timeout. Keep framework
            # shutdown bounded even if a third-party poll function blocks.
            thread.join(timeout=1.0)
            if not thread.is_alive():
                self._poll_thread = None

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            try:
                sample = self._poll()
                if sample is not None and not isinstance(sample, dict):
                    raise TypeError("poll callable must return a dictionary or None")
                with self._sample_lock:
                    self._latest_sample = copy.deepcopy(sample)
                    self._observed_at = time.time()
                    self._last_error = None
                    self._consecutive_failures = 0
            except Exception as exc:
                with self._sample_lock:
                    self._last_error = str(exc)
                    self._consecutive_failures += 1
            self._notify_completion()
            self._poll_stop.wait(self._interval)

    def _call_fn(self, request_id: int, content: Optional[Dict]) -> Optional[Dict]:
        with self._sample_lock:
            if self._observed_at is None:
                raise RuntimeError(self._last_error or "No telemetry sample is available")
            return {
                "sample": copy.deepcopy(self._latest_sample),
                "observed_at": self._observed_at,
            }

    def peek_status(self) -> Dict:
        status = super().peek_status()
        with self._sample_lock:
            observed_at = self._observed_at
            last_error = self._last_error
            failures = self._consecutive_failures
        initialized = (
            self._poll_thread is not None
            and self._poll_thread.is_alive()
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
