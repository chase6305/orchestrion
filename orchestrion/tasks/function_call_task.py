import concurrent.futures
import copy
import heapq
import threading
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

from orchestrion.retry import RetryPolicy
from orchestrion.schema import RequestSchema
from orchestrion.tasks.generic_task import GenericTask
from orchestrion.utils.logger import logger
from orchestrion.utils.types import PeekResponseResult, PeekResponseResultType


def _validate_max_result_count(max_result_count: int) -> None:
    if isinstance(max_result_count, bool) or not isinstance(max_result_count, int):
        raise TypeError("max_result_count must be an integer")
    if max_result_count == 0:
        raise ValueError("max_result_count must be positive or negative for unlimited")


class InPlaceFunctionCallTask(GenericTask):

    task_kind = "command"
    execution_mode = "in_place"
    supports_response_pruning = True

    def __init__(
        self, max_result_count: int = -1, metadata: Optional[Dict] = None
    ):
        """
        Initialize the in-place function call task.

        Args:
            max_result_count (int): Maximum number of results to keep in memory.
                                    If negative, unlimited.
        """
        super().__init__(metadata=metadata)
        _validate_max_result_count(max_result_count)
        self._result_map: Dict[int, Optional[Dict]] = (
            dict()
        )  # Stores results for each request_id
        self._latest_sent_request = -1  # Tracks the latest request_id sent
        self._accepted_request_floor = -1
        self._out_of_order_request_ids = set()
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
            if not self._reserve_request_id_locked(request_id):
                return False
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

    def _reserve_request_id_locked(self, request_id: int) -> bool:
        if self._request_id_is_reserved_locked(request_id):
            return False
        if request_id == self._accepted_request_floor + 1:
            self._accepted_request_floor = request_id
            while self._accepted_request_floor + 1 in self._out_of_order_request_ids:
                self._out_of_order_request_ids.remove(self._accepted_request_floor + 1)
                self._accepted_request_floor += 1
        else:
            self._out_of_order_request_ids.add(request_id)
        self._latest_sent_request = max(self._latest_sent_request, request_id)
        return True

    def _request_id_is_reserved_locked(self, request_id: int) -> bool:
        return (
            request_id <= self._accepted_request_floor
            or request_id in self._out_of_order_request_ids
        )

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
                    if self._request_id_is_reserved_locked(request_id)
                    else PeekResponseResultType.ErrorRequestNotSent
                )
                return PeekResponseResult(result_type, request_id=request_id)
            out = self._result_map[request_id]
        return PeekResponseResult(
            PeekResponseResultType.ResponseFound, request_id=request_id, content=out
        )

    def cancel_request(self, request_id: int) -> bool:
        return False

    def peek_status(self) -> Dict:
        with self._lock:
            return {
                "execution": "in_place",
                "latest_request_id": self._latest_sent_request,
                "retained_results": len(self._result_map),
            }

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

    task_kind = "command"
    execution_mode = "thread_pool"
    supports_cancellation = True
    supports_response_pruning = True

    def __init__(
        self, max_result_count: int = -1, metadata: Optional[Dict] = None
    ):
        """
        Initialize the threaded pool function call task.

        Args:
            max_result_count (int): Maximum number of results to keep in memory.
                                    If negative, unlimited.
        """
        super().__init__(metadata=metadata)
        _validate_max_result_count(max_result_count)
        self._executor_not_own: Optional[concurrent.futures.ThreadPoolExecutor] = (
            None  # External thread pool executor
        )
        self._future_map: Dict[int, concurrent.futures.Future] = (
            dict()
        )  # Maps request_id to Future objects
        self._latest_sent_request = -1  # Tracks the latest request_id sent
        self._accepted_request_floor = -1
        self._out_of_order_request_ids = set()
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
            if not self._reserve_request_id_locked(request_id):
                return False
            invoke_future = executor.submit(self._call_fn, request_id, content)
            invoke_future.add_done_callback(lambda _: self._notify_completion())
            self._future_map[request_id] = invoke_future
            if self._max_result_count > 0:
                heapq.heappush(self._request_id_pq, (request_id, request_id))
                while len(self._request_id_pq) > self._max_result_count:
                    pop_request_id, _ = heapq.heappop(self._request_id_pq)
                    pop_future = self._future_map.pop(pop_request_id, None)
                    if pop_future is not None:
                        pop_future.cancel()

        return True

    def _reserve_request_id_locked(self, request_id: int) -> bool:
        if self._request_id_is_reserved_locked(request_id):
            return False
        if request_id == self._accepted_request_floor + 1:
            self._accepted_request_floor = request_id
            while self._accepted_request_floor + 1 in self._out_of_order_request_ids:
                self._out_of_order_request_ids.remove(self._accepted_request_floor + 1)
                self._accepted_request_floor += 1
        else:
            self._out_of_order_request_ids.add(request_id)
        self._latest_sent_request = max(self._latest_sent_request, request_id)
        return True

    def _request_id_is_reserved_locked(self, request_id: int) -> bool:
        return (
            request_id <= self._accepted_request_floor
            or request_id in self._out_of_order_request_ids
        )

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
            request_was_sent = self._request_id_is_reserved_locked(request_id)
        if future is None:
            result_type = (
                PeekResponseResultType.ResponseReceivedButFlushed
                if request_was_sent
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

    def peek_status(self) -> Dict:
        with self._lock:
            futures = list(self._future_map.values())
            return {
                "execution": "thread_pool",
                "initialized": self._executor_not_own is not None,
                "latest_request_id": self._latest_sent_request,
                "retained_results": len(futures),
                "pending": sum(not future.done() for future in futures),
                "succeeded": sum(
                    future.done()
                    and not future.cancelled()
                    and future.exception() is None
                    for future in futures
                ),
                "failed": sum(
                    future.done()
                    and not future.cancelled()
                    and future.exception() is not None
                    for future in futures
                ),
                "cancelled": sum(future.cancelled() for future in futures),
            }

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


class CallableTask(ThreadedPoolFunctionCallTask):
    """Adapt ordinary callables into an asynchronous peripheral service."""

    def __init__(
        self,
        call: Callable[[int, Optional[Dict]], Optional[Dict]],
        status: Optional[Callable[[], Optional[Dict]]] = None,
        max_result_count: int = -1,
        retry_policy: Optional[RetryPolicy] = None,
        metadata: Optional[Dict] = None,
        request_schema: Optional[RequestSchema] = None,
    ):
        if not callable(call):
            raise TypeError("call must be callable")
        if status is not None and not callable(status):
            raise TypeError("status must be callable or None")
        if retry_policy is not None and not isinstance(retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be a RetryPolicy or None")
        if request_schema is not None and not isinstance(request_schema, RequestSchema):
            raise TypeError("request_schema must be a RequestSchema or None")
        super().__init__(max_result_count=max_result_count, metadata=metadata)
        self._call = call
        self._status = status
        self._retry_policy = retry_policy or RetryPolicy()
        self._request_schema = request_schema
        self._retry_lock = threading.Lock()
        self._latest_attempts = 0
        self._total_retries = 0
        self._last_retry_error: Optional[str] = None
        self._retry_stop_event = threading.Event()

    def describe(self) -> Dict:
        description = super().describe()
        description["retry"] = {
            "enabled": self._retry_policy.max_attempts > 1,
            "max_attempts": self._retry_policy.max_attempts,
            "delay": self._retry_policy.delay,
            "backoff": self._retry_policy.backoff,
            "max_delay": self._retry_policy.max_delay,
            "exception_types": [
                exception.__name__
                for exception in self._retry_policy.retry_exceptions
            ],
        }
        description["input_schema"] = (
            None
            if self._request_schema is None
            else self._request_schema.describe()
        )
        return copy.deepcopy(description)

    def initialize(self, **kwargs):
        self._retry_stop_event.clear()
        return super().initialize(**kwargs)

    def stop(self):
        self._retry_stop_event.set()
        return super().stop()

    def _call_fn(self, request_id: int, content: Optional[Dict]) -> Optional[Dict]:
        if self._request_schema is not None:
            content = self._request_schema.validate(content)
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            with self._retry_lock:
                self._latest_attempts = attempt
            try:
                return self._call(request_id, content)
            except concurrent.futures.CancelledError:
                raise
            except self._retry_policy.retry_exceptions as exc:
                with self._retry_lock:
                    self._last_retry_error = str(exc)
                if attempt >= self._retry_policy.max_attempts:
                    raise
                with self._retry_lock:
                    self._total_retries += 1
                self._notify_completion()
                delay = self._retry_policy.delay_before(attempt + 1)
                if delay and self._retry_stop_event.wait(delay):
                    raise concurrent.futures.CancelledError(
                        "Callable task stopped during retry backoff"
                    )
        raise AssertionError("unreachable")

    def peek_status(self) -> Dict:
        snapshot = super().peek_status()
        with self._retry_lock:
            snapshot.update(
                {
                    "latest_attempts": self._latest_attempts,
                    "total_retries": self._total_retries,
                    "last_retry_error": self._last_retry_error,
                }
            )
        if self._status is None:
            return snapshot
        custom = self._status()
        if custom is not None:
            if not isinstance(custom, dict):
                raise TypeError("status callable must return a dictionary or None")
            snapshot.update(custom)
        return snapshot
