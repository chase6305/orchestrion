"""Request lifecycle models shared by all Orchestrion tasks."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class RequestStatus(str, Enum):
    QUEUED = "queued"
    WAITING_FOR_MOVE = "waiting_for_move"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


@dataclass(frozen=True)
class RequestResult:
    request_id: int
    service_name: str
    status: RequestStatus
    content: Any = None
    error: Optional[str] = None
    associated_move_id: int = -1
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    priority: int = 0
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class TimelineEvent:
    request_id: int
    service_name: str
    status: RequestStatus
    timestamp: float
    message: Optional[str] = None

    @classmethod
    def now(
        cls,
        request_id: int,
        service_name: str,
        status: RequestStatus,
        message: Optional[str] = None,
    ) -> "TimelineEvent":
        return cls(request_id, service_name, status, time.time(), message)
