"""Bounded completed-result IDs with amortized cleanup of explicit deletions."""

import heapq
from typing import Optional


class ResultRetention:
    """Retain the largest IDs; callers supply locking and never reuse IDs."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._heap = []
        self._live = set()

    def add(self, request_id: int) -> Optional[int]:
        """Register a completed result and return an evicted ID, if any."""
        if request_id in self._live:
            return None
        if len(self._live) == self._capacity:
            while self._heap[0] not in self._live:
                heapq.heappop(self._heap)
            expired = heapq.heappushpop(self._heap, request_id)
            if expired != request_id:
                self._live.remove(expired)
                self._live.add(request_id)
            return expired
        self._live.add(request_id)
        heapq.heappush(self._heap, request_id)
        return None

    def discard(self, request_id: int) -> None:
        """Forget an ID without rebuilding the heap for every result removed."""
        self._live.discard(request_id)
        if not self._live:
            self._heap.clear()
        elif len(self._heap) > 2 * len(self._live) + 64:
            self._heap = list(self._live)
            heapq.heapify(self._heap)
