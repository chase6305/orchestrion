from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


class PeekResponseResultType(Enum):
    """
    Enum for possible response states when peeking for a task response.
    """

    ErrorUnknown = 0  # Unknown error occurred
    ErrorRequestNotSent = 1  # Request was not sent
    RequestSentNoResponse = 2  # Request sent but no response yet
    ResponseFound = 3  # Response found and available
    ResponseReceivedButFlushed = 4  # Response received but already flushed


@dataclass
class PeekResponseResult(object):
    result_type: PeekResponseResultType = PeekResponseResultType.ErrorUnknown
    request_id: int = -1
    content: Optional[Dict] = None

    @staticmethod
    def error_unknown():
        """
        Static method to return a default unknown error result.
        """
        return PeekResponseResult(
            result_type=PeekResponseResultType.ErrorUnknown, request_id=-1, content=None
        )
