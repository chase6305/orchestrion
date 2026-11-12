"""Response envelopes shared by legacy task adapters and TaskPilot."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class PeekResponseResultType(Enum):
    """Distinguish unsent, pending, available, failed and evicted responses."""

    ErrorUnknown = 0
    ErrorRequestNotSent = 1
    RequestSentNoResponse = 2
    ResponseFound = 3
    ResponseReceivedButFlushed = 4


@dataclass
class PeekResponseResult:
    """Adapter response; request ID -1 is unspecified for legacy backends."""

    result_type: PeekResponseResultType = PeekResponseResultType.ErrorUnknown
    request_id: int = -1
    content: Optional[Dict] = None
    error: Optional[str] = None
    cancelled: bool = False

    @staticmethod
    def error_unknown() -> "PeekResponseResult":
        """Return an error response without a request ID."""
        return PeekResponseResult()
