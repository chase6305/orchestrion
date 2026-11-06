"""Bounded SDK-thread progress delivery without one loop callback per message."""

import asyncio
import threading
from collections import deque
from typing import Any, Callable, Optional


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
        self._scheduled = False
        self._open = True
        self._dropped = 0
        self._unreported_drops = 0

    def post(self, snapshot: Any) -> None:
        with self._lock:
            if not self._open or self._loop.is_closed():
                return
            if self._pending is None:
                self._pending = deque()
            if len(self._pending) == self._max_pending:
                self._pending.popleft()
                self._dropped += 1
                self._unreported_drops += 1
            self._pending.append(snapshot)
            if not self._scheduled:
                self._scheduled = True
                try:
                    self._loop.call_soon_threadsafe(self._drain)
                except RuntimeError:
                    self._scheduled = False
                    self._pending.clear()
                    if not self._loop.is_closed():
                        raise

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
                batch = [pending.popleft() for _ in range(32)]
            dropped = self._unreported_drops
            self._unreported_drops = 0
        try:
            if dropped and self._overflow is not None:
                self._overflow(dropped)
            for value in batch:
                if self._open:
                    self._deliver(value)
        finally:
            with self._lock:
                if self._open and self._pending:
                    # Yield between batches so progress cannot monopolize the loop.
                    self._loop.call_soon(self._drain)
                else:
                    self._scheduled = False

    def close(self) -> int:
        """Called by the owner loop; flush accepted feedback before the final state."""
        with self._lock:
            if not self._open:
                return self._dropped
            self._open = False
            batch = self._pending or ()
            self._pending = None
            dropped = self._unreported_drops
            self._unreported_drops = 0
            deliver, overflow = self._deliver, self._overflow
            self._deliver = self._overflow = None
        if dropped and overflow is not None:
            overflow(dropped)
        for value in batch:
            deliver(value)
        return self._dropped

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pending": len(self._pending) if self._pending is not None else 0,
                "dropped": self._dropped,
            }
