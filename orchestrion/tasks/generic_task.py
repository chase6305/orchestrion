import copy
import json
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from orchestrion.utils.logger import logger
from orchestrion.utils.types import PeekResponseResult


class GenericTask(ABC):

    task_kind = "custom"
    execution_mode = "custom"
    supports_cancellation = False
    supports_response_pruning = False

    def __init__(self, metadata: Optional[Dict] = None):
        """
        Initialize the task instance.
        """
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None")
        try:
            json.dumps(metadata or {}, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-compatible") from exc
        self._metadata = copy.deepcopy(metadata or {})
        self._completion_callback: Optional[Callable[[], None]] = None

    def describe(self) -> Dict:
        """Return static, side-effect-free service discovery information."""
        return {
            "task_type": type(self).__name__,
            "kind": self.task_kind,
            "execution": self.execution_mode,
            "capabilities": {
                "cancellation": self.supports_cancellation,
                "response_pruning": self.supports_response_pruning,
                "status": type(self).peek_status is not GenericTask.peek_status,
            },
            "metadata": copy.deepcopy(getattr(self, "_metadata", {})),
        }

    def initialize(self, **kwargs):
        """
        Initialize the task with optional keyword arguments.
        """
        return None

    def stop(self):
        """
        Stop the task and release resources.
        """
        return None

    @abstractmethod
    def invoke_async(self, request_id: int, content: Optional[Dict] = None) -> bool:
        """
        Asynchronously invoke a request.

        Args:
        request_id (int): The request identifier.
            content (Optional[Dict]): Optional request content.
        Returns:
            bool: True if the request was accepted, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def peek_response(self, request_id: int) -> PeekResponseResult:
        """
        Peek for the response of a given request.

        Args:
            request_id (int): The request identifier.

        Returns:
            PeekResponseResult: The result of the peek operation.
        """
        raise NotImplementedError

    def peek_status(self) -> Optional[Dict]:
        """
        Peek for the current status of the task.

        Returns:
            Optional[Dict]: Status information if available, otherwise None.
        """
        return None

    def cancel_request(self, request_id: int) -> bool:
        """Attempt to cancel a running request.

        Tasks that support cancellation should override this method.
        """
        return False

    def forget_response(self, request_id: int) -> bool:
        """Release a terminal response retained by this task, if supported."""
        return False

    def set_completion_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Register a callback invoked whenever a request may have completed."""
        self._completion_callback = callback

    def _notify_completion(self) -> None:
        if self._completion_callback is not None:
            try:
                self._completion_callback()
            except Exception:
                logger.exception("Task completion callback failed")
