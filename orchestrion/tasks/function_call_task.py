import concurrent.futures
import heapq
import threading
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from orchestrion.tasks.generic_task import GenericTask
from orchestrion.utils.logger import logger
from orchestrion.utils.types import PeekResponseResult, PeekResponseResultType


def _validate_max_result_count(max_result_count: int) -> None:
    if isinstance(max_result_count, bool) or not isinstance(max_result_count, int):
        raise TypeError("max_result_count must be an integer")
    if max_result_count == 0:
        raise ValueError("max_result_count must be positive or negative for unlimited")


class InPlaceFunctionCallTask(GenericTask):

    def __init__(self, max_result_count: int = -1):
        """
        Initialize the in-place function call task.

        Args:
            max_result_count (int): Maximum number of results to keep in memory.
                                    If negative, unlimited.
        """
        super().__init__()
        _validate_max_result_count(max_result_count)
        self._result_map: Dict[int, Optional[Dict]] = (
            dict()
        )  # Stores results for each request_id
        self._latest_sent_request = -1  # Tracks the latest request_id sent
        self._max_result_count = max_result_count  # Maximum number of results to retain
        self._lock = threading.Lock()
        # Priority queue to manage request_ids for result eviction when max_result_count is set
        self._request_id_pq: Optional[List[Tuple[int, int]]] = (
            None if max_result_count < 0 else list()
        )

    @abstractmethod
    def _call_fn(self, request_id: int, content: Optional[Dict]) -> Optional[Dict]:
        """
        Internal function to process the request. Should be overridden by subclasses.

        Args:
            request_id (int): The request identifier.
            content (Optional[Dict]): Optional request content.

        Returns:
            Optional[Dict]: The result of the function call, or None if not implemented.
        """
        raise NotImplementedError

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
        with self._lock:
            if request_id <= self._latest_sent_request:
                return False
            self._latest_sent_request = request_id
        out = self._call_fn(request_id, content)

        with self._lock:
            self._result_map[request_id] = out
            if self._max_result_count > 0:
                heapq.heappush(self._request_id_pq, (request_id, request_id))
                while len(self._request_id_pq) > self._max_result_count:
                    pop_request_id, _ = heapq.heappop(self._request_id_pq)
                    self._result_map.pop(pop_request_id, None)

        self._notify_completion()

        return True

    def peek_response(self, request_id: int) -> PeekResponseResult:
        """
        Peek for the response of a given request.

        Args:
            request_id (int): The request identifier.

        Returns:
            PeekResponseResult: The result of the peek operation. Returns error if not found.
        """
        with self._lock:
            if request_id not in self._result_map:
                result_type = (
                    PeekResponseResultType.ResponseReceivedButFlushed
                    if request_id <= self._latest_sent_request
                    else PeekResponseResultType.ErrorRequestNotSent
                )
                return PeekResponseResult(result_type, request_id=request_id)
            out = self._result_map[request_id]
        return PeekResponseResult(
            PeekResponseResultType.ResponseFound, request_id=request_id, content=out
        )

    def cancel_request(self, request_id: int) -> bool:
        return False

    def forget_response(self, request_id: int) -> bool:
        with self._lock:
            if request_id not in self._result_map:
                return False
            self._result_map.pop(request_id)
            if self._request_id_pq is not None:
                self._request_id_pq = [
                    entry for entry in self._request_id_pq if entry[0] != request_id
                ]
                heapq.heapify(self._request_id_pq)
            return True


class ThreadedPoolFunctionCallTask(GenericTask):

    def __init__(self, max_result_count: int = -1):
        """
        Initialize the threaded pool function call task.

        Args:
            max_result_count (int): Maximum number of results to keep in memory.
                                    If negative, unlimited.
        """
        super().__init__()
        _validate_max_result_count(max_result_count)
        self._executor_not_own: Optional[concurrent.futures.ThreadPoolExecutor] = (
            None  # External thread pool executor
        )
        self._future_map: Dict[int, concurrent.futures.Future] = (
            dict()
        )  # Maps request_id to Future objects
        self._latest_sent_request = -1  # Tracks the latest request_id sent
        self._max_result_count = max_result_count  # Maximum number of results to retain
        self._lock = threading.Lock()
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
        if executor is None:
            raise ValueError("ThreadedPoolFunctionCallTask requires an executor")
        with self._lock:
            self._executor_not_own = executor

    def stop(self):
        """
        Stop the task and cancel all pending futures. Does NOT destruct the executor.
        """
        with self._lock:
            futures = list(self._future_map.values())
            self._executor_not_own = None
        for future in futures:
            future.cancel()

    @abstractmethod
    def _call_fn(self, request_id: int, content: Optional[Dict]) -> Optional[Dict]:
        """
        Internal function to process the request. Should be overridden by subclasses.

        Args:
            request_id (int): The request identifier.
            content (Optional[Dict]): Optional request content.

        Returns:
            Optional[Dict]: The result of the function call, or None if not implemented.
        """
        raise NotImplementedError

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
        with self._lock:
            executor = self._executor_not_own
            if executor is None:
                raise RuntimeError("Task must be initialized before use")
            if request_id <= self._latest_sent_request:
                return False
            invoke_future = executor.submit(self._call_fn, request_id, content)
            invoke_future.add_done_callback(lambda _: self._notify_completion())
            self._latest_sent_request = request_id
            self._future_map[request_id] = invoke_future
            if self._max_result_count > 0:
                heapq.heappush(self._request_id_pq, (request_id, request_id))
                while len(self._request_id_pq) > self._max_result_count:
                    pop_request_id, _ = heapq.heappop(self._request_id_pq)
                    pop_future = self._future_map.pop(pop_request_id, None)
                    if pop_future is not None:
                        pop_future.cancel()

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
        with self._lock:
            future = self._future_map.get(request_id)
            latest_sent_request = self._latest_sent_request
        if future is None:
            result_type = (
                PeekResponseResultType.ResponseReceivedButFlushed
                if request_id <= latest_sent_request
                else PeekResponseResultType.ErrorRequestNotSent
            )
            return PeekResponseResult(result_type, request_id=request_id)
        if future.done():
            if future.cancelled():
                return PeekResponseResult(
                    PeekResponseResultType.ErrorUnknown,
                    request_id=request_id,
                    content=None,
                )
            else:
                exception = future.exception()
                if exception is not None:
                    logger.error("Request %s failed: %s", request_id, exception)
                    return PeekResponseResult(
                        PeekResponseResultType.ErrorUnknown,
                        request_id=request_id,
                        error=str(exception),
                    )
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

    def cancel_request(self, request_id: int) -> bool:
        with self._lock:
            future = self._future_map.get(request_id)
        return future is not None and future.cancel()

    def forget_response(self, request_id: int) -> bool:
        with self._lock:
            future = self._future_map.get(request_id)
            if future is None or not future.done():
                return False
            self._future_map.pop(request_id)
            if self._request_id_pq is not None:
                self._request_id_pq = [
                    entry for entry in self._request_id_pq if entry[0] != request_id
                ]
                heapq.heapify(self._request_id_pq)
            return True
