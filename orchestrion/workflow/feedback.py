"""Bounded SDK-thread progress delivery without one loop callback per message."""

import asyncio
import threading
from collections import deque
from typing import Any, Callable, Optional

from orchestrion.utils.errors import exception_description
from orchestrion.utils.logger import log_error


def deliver_feedback(callback: Callable[[Any], None], value: Any) -> bool:
    """Optional progress cannot decide whether a physical operation has stopped."""
    try:
        callback(value)
    except Exception as exc:
        log_error("Workflow feedback callback failed: %s", exception_description(exc))
        return False
    return True


class FeedbackChannel:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        deliver: Callable[[Any], None],
        overflow: Optional[Callable[[int], None]],
        max_pending: int,
    ) -> None:
        self._loop = loop
        self._deliver = deliver
        self._overflow = overflow
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._pending = None
        self._draining = None
        self._scheduled = False
        self._open = True
        self._dropped = 0
        self._unreported_drops = 0
        self._notification_failed = False

    def _schedule_locked(self, *, threadsafe: bool):
        """Return retired data/errors to release and format after unlocking."""
        self._scheduled = True
        try:
            schedule = self._loop.call_soon_threadsafe if threadsafe else self._loop.call_soon
            schedule(self._drain)
        except Exception as exc:
            abandoned, self._pending = self._pending, None
            lost = len(abandoned) if abandoned is not None else 0
            self._dropped += lost
            self._unreported_drops += lost
            self._scheduled = False
            should_log = not self._notification_failed and not self._loop.is_closed()
            self._notification_failed = True
            return abandoned, exc, should_log
        self._notification_failed = False
        return None, None, False

    @staticmethod
    def _log_notification_failure(error, should_log: bool) -> None:
        if should_log:
            log_error("Workflow feedback notification failed: %s", exception_description(error))

    def post(self, snapshot: Any) -> None:
        retired = abandoned = error = None
        should_log = False
        with self._lock:
            if not self._open or self._loop.is_closed():
                return
            if self._pending is None:
                self._pending = deque()
            if len(self._pending) == self._max_pending:
                retired = self._pending.popleft()
                self._dropped += 1
                self._unreported_drops += 1
            self._pending.append(snapshot)
            if not self._scheduled:
                abandoned, error, should_log = self._schedule_locked(threadsafe=True)
        # Scalar subclasses can own SDK finalizers that query this channel.
        del retired, abandoned
        self._log_notification_failure(error, should_log)

    def _deliver_one(self, callback, value) -> None:
        if not deliver_feedback(callback, value):
            with self._lock:
                self._dropped += 1

    def _drain(self) -> None:
        with self._lock:
            if not self._open:
                return
            pending = self._pending
            if len(pending) <= 32:
                # Detach the final batch so producers cannot mutate it during delivery.
                batch = pending
                self._pending = None
            else:
                batch = deque(pending.popleft() for _ in range(32))
            self._draining = batch
            dropped = self._unreported_drops
            self._unreported_drops = 0
        try:
            if dropped and self._overflow is not None:
                deliver_feedback(self._overflow, dropped)
            while batch and self._open:
                self._deliver_one(self._deliver, batch.popleft())
        finally:
            abandoned = error = None
            should_log = False
            with self._lock:
                if self._draining is batch:
                    self._draining = None
                if self._open and self._pending:
                    # Yield between batches so progress cannot monopolize the loop.
                    abandoned, error, should_log = self._schedule_locked(threadsafe=False)
                else:
                    self._scheduled = False
            del abandoned
            self._log_notification_failure(error, should_log)

    def close(self) -> int:
        """Called by the owner loop; flush accepted feedback before the final state."""
        with self._lock:
            if not self._open:
                return self._dropped
            self._open = False
            draining, self._draining = self._draining, None
            batch = self._pending or ()
            self._pending = None
            dropped = self._unreported_drops
            self._unreported_drops = 0
            deliver, overflow = self._deliver, self._overflow
            self._deliver = self._overflow = None
        if dropped and overflow is not None:
            deliver_feedback(overflow, dropped)
        if draining is not None:
            while draining:
                self._deliver_one(deliver, draining.popleft())
        for value in batch:
            self._deliver_one(deliver, value)
        return self._dropped

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pending": len(self._pending) if self._pending is not None else 0,
                "dropped": self._dropped,
            }
