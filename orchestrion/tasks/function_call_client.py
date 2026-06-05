import heapq
import concurrent.futures
from typing import Dict, Optional, Tuple, List

from orchestrion.utils.types import PeekResponseResult, PeekResponseResultType
from orchestrion.tasks.generic_task import GenericTask
from orchestrion.utils.logger import logger


class InPlaceFunctionCallTask(GenericTask):

    def __init__(self, max_result_count: int = -1):
        """
        Initialize the in-place function call task.

        Args:
            max_result_count (int): Maximum number of results to keep in memory.
                                    If negative, unlimited.
        """
        super().__init__()
        self._result_map: Dict[int, Optional[Dict]] = (
            dict()
        )  # Stores results for each request_id
        self._lastest_sent_request = -1  # Tracks the latest request_id sent
        self._max_result_count = max_result_count  # Maximum number of results to retain
        # Priority queue to manage request_ids for result eviction when max_result_count is set
        self._request_id_pq: Optional[List[Tuple[int, int]]] = (
            None if max_result_count < 0 else list()
        )

    def _call_fn(self, request_id: int, content: Optional[Dict]) -> Optional[Dict]:
        """
        Internal function to process the request. Should be overridden by subclasses.、

        Args:
            request_id (int): The request identifier.
            content (Optional[Dict]): Optional request content.

        Returns:
            Optional[Dict]: The result of the function call, or None if not implemented.
        """
        return None

    def invoke_async(self, request_id: int, content: Optional[Dict] = None) -> bool:
        """
        Asynchronously invoke a function call and store the result.

        Args:
            request_id (int): The request identifier.
            content (Optional[Dict]): Optional request content.

        Returns:
            bool: True if the request was accepted and processed, False otherwise.
        """
        # Check if the request_id is valid and not duplicated
        if request_id <= self._lastest_sent_request or request_id in self._result_map:
            return False

        # Update the latest sent request
        self._lastest_sent_request = request_id
        out = self._call_fn(request_id, content)

        # Remove oldest results if exceeding max_result_count
        if self._max_result_count > 0:
            while len(self._request_id_pq) > self._max_result_count:
                pop_request_id, _ = heapq.heappop(self._request_id_pq)
                assert pop_request_id in self._result_map
                self._result_map.pop(pop_request_id)

        # Store the result for this request_id
        self._result_map[request_id] = out
        if self._max_result_count > 0:
            heapq.heappush(self._request_id_pq, (request_id, request_id))

        return True

    def peek_response(self, request_id: int) -> PeekResponseResult:
        """
        Peek for the response of a given request.

        Args:
            request_id (int): The request identifier.

        Returns:
            PeekResponseResult: The result of the peek operation. Returns error if not found.
        """
        if request_id not in self._result_map:
            return PeekResponseResult.error_unknown()

        # Return the stored result
        out = self._result_map[request_id]
        return PeekResponseResult(
            PeekResponseResultType.ResponseFound, request_id=request_id, content=out
        )


class ThreadedPoolFunctionCallTask(GenericTask):

    def __init__(self, max_result_count: int = -1):
        """
        Initialize the threaded pool function call task.

        Args:
            max_result_count (int): Maximum number of results to keep in memory.
                                    If negative, unlimited.
        """
        super().__init__()
        self._executor_not_own: Optional[concurrent.futures.ThreadPoolExecutor] = (
            None  # External thread pool executor
        )
        self._future_map: Dict[int, concurrent.futures.Future] = (
            dict()
        )  # Maps request_id to Future objects
        self._lastest_sent_request = -1  # Tracks the latest request_id sent
        self._max_result_count = max_result_count  # Maximum number of results to retain
        # Priority queue to manage request_ids for result eviction when max_result_count is set
        self._request_id_pq: Optional[List[Tuple[int, int]]] = (
            None if max_result_count < 0 else list()
        )

    def initialize(self, **kwargs):
        """
        Initialize the task with an external thread pool executor.

        Args:
            executor: ThreadPoolExecutor instance (must be provided in kwargs).
        """
        # Does NOT init executor, maintained outside
        executor = kwargs.get("executor", None)
        assert executor is not None
        self._executor_not_own = executor

    def stop(self):
        """
        Stop the task and cancel all pending futures. Does NOT destruct the executor.
        """
        for _, v in self._future_map.items():
            v.cancel()

    def _call_fn(self, request_id: int, content: Optional[Dict]) -> Optional[Dict]:
        """
        Internal function to process the request. Should be overridden by subclasses.

        Args:
            request_id (int): The request identifier.
            content (Optional[Dict]): Optional request content.

        Returns:
            Optional[Dict]: The result of the function call, or None if not implemented.
        """
        return None

    def invoke_async(self, request_id: int, content: Optional[Dict] = None) -> bool:
        """
        Asynchronously invoke a function call using a thread pool and store the future.

        Args:
            request_id (int): The request identifier.
            content (Optional[Dict]): Optional request content.

        Returns:
            bool: True if the request was accepted and processed, False otherwise.
        """
        # Check if the request_id is valid and not duplicated
        if request_id <= self._lastest_sent_request or request_id in self._future_map:
            return False

        # Update the latest sent request
        self._lastest_sent_request = request_id
        invoke_future = self._executor_not_own.submit(
            self._call_fn, request_id, content
        )

        # Remove oldest results if exceeding max_result_count
        if self._max_result_count > 0:
            while len(self._request_id_pq) > self._max_result_count:
                pop_request_id, _ = heapq.heappop(self._request_id_pq)
                assert pop_request_id in self._future_map
                pop_future = self._future_map.pop(pop_request_id)
                pop_future.cancel()

        # Store the future for this request_id
        self._future_map[request_id] = invoke_future
        if self._max_result_count > 0:
            heapq.heappush(self._request_id_pq, (request_id, request_id))

        return True

    def peek_response(self, request_id: int) -> PeekResponseResult:
        """
        Peek for the response of a given request.

        Args:
            request_id (int): The request identifier.

        Returns:
            PeekResponseResult: The result of the peek operation.
                                Returns error if not found or not ready.
        """
        if request_id not in self._future_map:
            return PeekResponseResult(
                PeekResponseResultType.ErrorRequestNotSent,
                request_id=request_id,
                content=None,
            )

        # Get the future object for this request
        future: concurrent.futures.Future = self._future_map[request_id]
        if future.done():
            if future.cancelled():
                return PeekResponseResult(
                    PeekResponseResultType.ErrorUnknown,
                    request_id=request_id,
                    content=None,
                )
            else:
                out = future.result()
                return PeekResponseResult(
                    PeekResponseResultType.ResponseFound,
                    request_id=request_id,
                    content=out,
                )
        else:
            return PeekResponseResult(
                PeekResponseResultType.RequestSentNoResponse,
                request_id=request_id,
                content=None,
            )
