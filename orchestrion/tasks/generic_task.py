from abc import abstractmethod
from typing import Dict, Optional
from orchestrion.utils.types import PeekResponseResult, PeekResponseResultType


class GenericTask(object):

    def __init__(self):
        """
        Initialize the task instance.
        """
        pass

    def initialize(self, **kwargs):
        """
        Initialize the task with optional keyword arguments.
        """
        pass

    def stop(self):
        """
        Stop the task and release resources.
        """
        pass

    @abstractmethod
    def invoke_async(
        self, request_id: int, move_id: int, content: Optional[Dict] = None
    ) -> bool:
        """
        Asynchronously invoke a request.

        Args:
        request_id (int): The request identifier.
            move_id (int): The movement identifier (if applicable).
            content (Optional[Dict]): Optional request content.
        Returns:
            bool: True if the request was accepted, False otherwise.
        """
        pass

    @abstractmethod
    def peek_response(self, request_id: int) -> PeekResponseResult:
        """
        Peek for the response of a given request.

        Args:
            request_id (int): The request identifier.

        Returns:
            PeekResponseResult: The result of the peek operation.
        """
        return PeekResponseResult.error_unknown()

    def peek_status(self) -> Optional[Dict]:
        """
        Peek for the current status of the task.

        Returns:
            Optional[Dict]: Status information if available, otherwise None.
        """
        return None
