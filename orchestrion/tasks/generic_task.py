import copy
import json
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from orchestrion.utils.logger import logger
from orchestrion.utils.types import PeekResponseResult


class GenericTask(ABC):
    """Adapter hooks for request execution, status and lifecycle management."""

    task_kind = "custom"
    execution_mode = "custom"
    supports_cancellation = False
    supports_response_pruning = False

    def __init__(self, metadata: Optional[Dict] = None):
        """Snapshot optional JSON metadata for service discovery."""
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None")
        metadata_snapshot = copy.deepcopy({} if metadata is None else metadata)
        if not isinstance(metadata_snapshot, dict):
            raise TypeError("copied metadata must be a dictionary")
        try:
            json.dumps(metadata_snapshot, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-compatible") from exc
        self._metadata = metadata_snapshot
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

    def initialize(self, **kwargs) -> None:
        """Acquire adapter resources; subclasses may accept configuration keywords."""

    def stop(self) -> None:
        """Release adapter resources."""

    @abstractmethod
    def invoke_async(self, request_id: int, content: Optional[Dict] = None) -> bool:
        """Submit a request and return whether it was accepted."""
        raise NotImplementedError

    @abstractmethod
    def peek_response(self, request_id: int) -> PeekResponseResult:
        """Read request progress or its response without waiting."""
        raise NotImplementedError

    def peek_status(self) -> Optional[Dict]:
        """Return status information, or None when status queries are unsupported."""
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
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        self._completion_callback = callback

    def _notify_completion(self) -> None:
        callback = self._completion_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.exception("Task completion callback failed")
