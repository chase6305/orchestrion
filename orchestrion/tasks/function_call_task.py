import concurrent.futures
import copy
import threading
from abc import abstractmethod
from functools import partial
from typing import Callable, Dict, Optional

from orchestrion.retry import RetryPolicy
from orchestrion.schema import RequestSchema
from orchestrion.tasks.generic_task import GenericTask
from orchestrion.utils.errors import exception_message
from orchestrion.utils.logger import logger
from orchestrion.utils.request_history import RequestHistory
from orchestrion.utils.result_retention import ResultRetention
from orchestrion.utils.types import PeekResponseResult, PeekResponseResultType


def _validate_max_result_count(max_result_count: int) -> None:
    if isinstance(max_result_count, bool) or not isinstance(max_result_count, int):
        raise TypeError("max_result_count must be an integer")
    if max_result_count == 0:
        raise ValueError("max_result_count must be positive or negative for unlimited")


def _validate_request_id(request_id: int) -> None:
    """Reject IDs that can alias valid requests or corrupt ordering state."""
    if isinstance(request_id, bool) or not isinstance(request_id, int):
        raise TypeError("request_id must be an integer")
    if request_id < 0:
        raise ValueError("request_id must be non-negative")


class InPlaceFunctionCallTask(GenericTask):
    """Run each call on the invoking thread and retain its response."""

    task_kind = "command"
    execution_mode = "in_place"
    supports_response_pruning = True

    def __init__(
        self, max_result_count: int = -1, metadata: Optional[Dict] = None
    ):
        """Retain at most max_result_count responses; a negative value is unlimited."""
        super().__init__(metadata=metadata)
        _validate_max_result_count(max_result_count)
        self._result_map: Dict[int, PeekResponseResult] = {}
        self._latest_request_id = -1
        self._request_history = RequestHistory()
        self._pending_request_ids = set()
        self._lock = threading.Lock()
        self._retention = (
            None if max_result_count < 0 else ResultRetention(max_result_count)
        )

    @abstractmethod
    def _call_fn(self, request_id: int, content: Optional[Dict]) -> Optional[Dict]:
        """Execute the request and return its response."""
        raise NotImplementedError

    def invoke_async(self, request_id: int, content: Optional[Dict] = None) -> bool:
        """Execute inline and retain the response; reject duplicate request IDs."""
        _validate_request_id(request_id)
        with self._lock:
            if not self._request_history.add(request_id):
                return False
            self._latest_request_id = max(self._latest_request_id, request_id)
            self._pending_request_ids.add(request_id)
        try:
            response_content = self._call_fn(request_id, content)
        except Exception as exc:
            self._store_response(
                PeekResponseResult(
                    PeekResponseResultType.ErrorUnknown,
                    request_id=request_id,
                    error=exception_message(exc),
                )
            )
            raise
        self._store_response(
            PeekResponseResult(
                PeekResponseResultType.ResponseFound,
                request_id=request_id,
                content=response_content,
            )
        )
        return True

    def _store_response(self, response: PeekResponseResult) -> None:
        request_id = response.request_id
        with self._lock:
            self._pending_request_ids.discard(request_id)
            self._result_map[request_id] = response
            if self._retention is not None:
                expired_id = self._retention.add(request_id)
                if expired_id is not None:
                    self._result_map.pop(expired_id, None)

        self._notify_completion()

    def peek_response(self, request_id: int) -> PeekResponseResult:
        """Return a response snapshot or distinguish pending, flushed and unsent IDs."""
        _validate_request_id(request_id)
        with self._lock:
            if request_id in self._pending_request_ids:
                return PeekResponseResult(
                    PeekResponseResultType.RequestSentNoResponse, request_id=request_id
                )
            if request_id not in self._result_map:
                result_type = (
                    PeekResponseResultType.ResponseReceivedButFlushed
                    if request_id in self._request_history
                    else PeekResponseResultType.ErrorRequestNotSent
                )
                return PeekResponseResult(result_type, request_id=request_id)
            response = self._result_map[request_id]
        # A retained response is replaced or removed, never updated by the task.
        # User-defined deepcopy hooks must not run while holding the task lock.
        return copy.deepcopy(response)

    def cancel_request(self, request_id: int) -> bool:
        """Inline calls cannot be interrupted."""
        _validate_request_id(request_id)
        return False

    def peek_status(self) -> Dict:
        with self._lock:
            return {
                "execution": "in_place",
                "latest_request_id": self._latest_request_id,
                "retained_results": len(self._result_map),
            }

    def forget_response(self, request_id: int) -> bool:
        _validate_request_id(request_id)
        with self._lock:
            if request_id not in self._result_map:
                return False
            self._result_map.pop(request_id)
            if self._retention is not None:
                self._retention.discard(request_id)
            return True


class ThreadedPoolFunctionCallTask(GenericTask):
    """Submit calls to a borrowed executor and retain their futures."""

    task_kind = "command"
    execution_mode = "thread_pool"
    supports_cancellation = True
    supports_response_pruning = True

    def __init__(
        self, max_result_count: int = -1, metadata: Optional[Dict] = None
    ):
        """Bound completed responses while retaining every pending future."""
        super().__init__(metadata=metadata)
        _validate_max_result_count(max_result_count)
        self._borrowed_executor: Optional[concurrent.futures.ThreadPoolExecutor] = (
            None
        )
        self._future_map: Dict[int, concurrent.futures.Future] = {}
        self._pending_futures: Dict[int, concurrent.futures.Future] = {}
        self._outcomes: Dict[int, str] = {}
        self._outcome_counts = dict.fromkeys(("succeeded", "failed", "cancelled"), 0)
        self._latest_request_id = -1
        self._request_history = RequestHistory()
        self._lock = threading.Lock()
        self._retention = (
            None if max_result_count < 0 else ResultRetention(max_result_count)
        )

    def initialize(self, **kwargs):
        """Bind a caller-owned executor supplied as the executor keyword argument."""
        executor = kwargs.get("executor")
        if executor is None:
            raise ValueError("ThreadedPoolFunctionCallTask requires an executor")
        if not callable(getattr(executor, "submit", None)):
            raise TypeError("executor must provide a callable submit method")
        with self._lock:
            self._borrowed_executor = executor

    def stop(self):
        """Cancel queued futures and detach from the caller-owned executor."""
        with self._lock:
            futures = list(self._pending_futures.values())
            self._borrowed_executor = None
        for future in futures:
            future.cancel()

    @abstractmethod
    def _call_fn(self, request_id: int, content: Optional[Dict]) -> Optional[Dict]:
        """Execute the request and return its response."""
        raise NotImplementedError

    def invoke_async(self, request_id: int, content: Optional[Dict] = None) -> bool:
        """Submit once; a failed executor submission leaves the request ID reusable."""
        _validate_request_id(request_id)
        with self._lock:
            executor = self._borrowed_executor
            if executor is None:
                raise RuntimeError("Task must be initialized before use")
            if request_id in self._request_history:
                return False
            # Commit the ID only after submission succeeds. Holding the task lock
            # prevents another caller reserving it and avoids copying the entire
            # sparse-ID history on every submission.
            future = self._submit_call(executor, request_id, content)
            self._request_history.add(request_id)
            self._latest_request_id = max(self._latest_request_id, request_id)
            self._future_map[request_id] = future
            self._pending_futures[request_id] = future
        # Register outside the lock: add_done_callback() invokes synchronously
        # when a very fast future has already completed.
        future.add_done_callback(partial(self._on_future_done, request_id))

        return True

    def _submit_call(
        self, executor: concurrent.futures.ThreadPoolExecutor,
        request_id: int, content: Optional[Dict],
    ) -> concurrent.futures.Future:
        """Prepare a call while submission is protected by the task lock."""
        return executor.submit(self._call_fn, request_id, content)

    def _on_future_done(
        self, request_id: int, future: concurrent.futures.Future
    ) -> None:
        with self._lock:
            self._record_outcome_locked(request_id, future)
            failed = self._outcomes.get(request_id) == "failed"
            if (
                self._retention is not None
                and self._future_map.get(request_id) is future
            ):
                # Only completed futures enter the eviction heap. Pending work
                # never needs to be scanned, sorted, or cancelled to retain K results.
                expired_id = self._retention.add(request_id)
                if expired_id is not None:
                    self._remove_future_locked(expired_id)
        # Logging and SDK exception formatters may call arbitrary application
        # code. Keep them outside the task lock, and always notify observers.
        try:
            if failed:
                logger.error(
                    "Request %s failed: %s",
                    request_id, exception_message(future.exception()),
                )
        finally:
            self._notify_completion()

    def _record_outcome_locked(
        self, request_id: int, future: concurrent.futures.Future
    ) -> None:
        """Account for a completed future once, including before its callback runs."""
        if self._pending_futures.get(request_id) is not future:
            return
        try:
            error = future.exception()
        except concurrent.futures.CancelledError:
            outcome = "cancelled"
        else:
            outcome = "failed" if error is not None else "succeeded"
        self._pending_futures.pop(request_id)
        self._outcomes[request_id] = outcome
        self._outcome_counts[outcome] += 1

    def _remove_future_locked(self, request_id: int) -> None:
        self._future_map.pop(request_id, None)
        self._pending_futures.pop(request_id, None)
        outcome = self._outcomes.pop(request_id, None)
        if outcome is not None:
            self._outcome_counts[outcome] -= 1

    def peek_response(self, request_id: int) -> PeekResponseResult:
        """Return a response snapshot or distinguish pending, flushed and unsent IDs."""
        _validate_request_id(request_id)
        with self._lock:
            future = self._future_map.get(request_id)
            request_was_sent = request_id in self._request_history
        if future is None:
            result_type = (
                PeekResponseResultType.ResponseReceivedButFlushed
                if request_was_sent
                else PeekResponseResultType.ErrorRequestNotSent
            )
            return PeekResponseResult(result_type, request_id=request_id)
        if not future.done():
            return PeekResponseResult(
                PeekResponseResultType.RequestSentNoResponse,
                request_id=request_id,
            )
        if future.cancelled():
            return PeekResponseResult(
                PeekResponseResultType.ErrorUnknown, request_id=request_id
            )
        exception = future.exception()
        if exception is not None:
            return PeekResponseResult(
                PeekResponseResultType.ErrorUnknown,
                request_id=request_id,
                error=exception_message(exception),
            )
        return PeekResponseResult(
            PeekResponseResultType.ResponseFound,
            request_id=request_id,
            content=copy.deepcopy(future.result()),
        )

    def cancel_request(self, request_id: int) -> bool:
        """Cancel a retained future only if its execution has not started."""
        _validate_request_id(request_id)
        with self._lock:
            future = self._future_map.get(request_id)
        return future is not None and future.cancel()

    def peek_status(self) -> Dict:
        """Count retained outcomes without revisiting already completed futures."""
        with self._lock:
            # Future completion precedes callback dispatch. Reconcile only pending
            # entries so a delayed callback does not make the snapshot stale.
            for request_id, future in tuple(self._pending_futures.items()):
                if future.done():
                    self._record_outcome_locked(request_id, future)
            return {
                "execution": "thread_pool",
                "initialized": self._borrowed_executor is not None,
                "latest_request_id": self._latest_request_id,
                "retained_results": len(self._future_map),
                "pending": len(self._pending_futures),
                **self._outcome_counts,
            }

    def forget_response(self, request_id: int) -> bool:
        _validate_request_id(request_id)
        with self._lock:
            future = self._future_map.get(request_id)
            if future is None or not future.done():
                return False
            self._remove_future_locked(request_id)
            if self._retention is not None:
                self._retention.discard(request_id)
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
        if request_schema is not None and not isinstance(
            request_schema, RequestSchema
        ):
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
        self._retry_lifecycle_lock = threading.RLock()
        self._call_context = threading.local()

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
        with self._retry_lifecycle_lock:
            super().initialize(**kwargs)
            if self._retry_stop_event.is_set():
                self._retry_stop_event = threading.Event()

    def stop(self):
        with self._retry_lifecycle_lock:
            self._retry_stop_event.set()
            return super().stop()

    def invoke_async(self, request_id: int, content: Optional[Dict] = None) -> bool:
        with self._retry_lifecycle_lock:
            return super().invoke_async(request_id, content)

    def _submit_call(
        self, executor: concurrent.futures.ThreadPoolExecutor,
        request_id: int, content: Optional[Dict],
    ) -> concurrent.futures.Future:
        # Bind the lifecycle token at submission, including executor-queued work.
        return executor.submit(
            self._call_in_lifecycle, request_id, content, self._retry_stop_event
        )

    def _call_in_lifecycle(
        self, request_id: int, content: Optional[Dict], stop_event: threading.Event
    ) -> Optional[Dict]:
        previous_stop_event = getattr(self._call_context, "stop_event", None)
        self._call_context.stop_event = stop_event
        try:
            # Preserve the inherited extension hook for CallableTask subclasses.
            return self._call_fn(request_id, content)
        finally:
            if previous_stop_event is None:
                del self._call_context.stop_event
            else:
                self._call_context.stop_event = previous_stop_event

    def _call_fn(self, request_id: int, content: Optional[Dict]) -> Optional[Dict]:
        stop_event = getattr(self._call_context, "stop_event", self._retry_stop_event)
        return self._call_with_retries(request_id, content, stop_event)

    def _call_with_retries(
        self, request_id: int, content: Optional[Dict], stop_event: threading.Event
    ) -> Optional[Dict]:
        if stop_event.is_set():
            raise concurrent.futures.CancelledError("Callable task stopped")
        if self._request_schema is not None:
            content = self._request_schema.validate(content)
        elif self._retry_policy.max_attempts > 1:
            content = copy.deepcopy(content)
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            if stop_event.is_set():
                raise concurrent.futures.CancelledError("Callable task stopped")
            attempt_content = content
            if self._retry_policy.max_attempts > 1:
                # Failed SDK calls may mutate their arguments. Preserve the
                # baseline and validate the exact copy sent on each attempt.
                attempt_content = (
                    self._request_schema.validate(content)
                    if self._request_schema is not None
                    else copy.deepcopy(content)
                )
            if stop_event.is_set():
                raise concurrent.futures.CancelledError("Callable task stopped")
            with self._retry_lock:
                self._latest_attempts = attempt
            try:
                return self._call(request_id, attempt_content)
            except concurrent.futures.CancelledError:
                raise
            except self._retry_policy.retry_exceptions as exc:
                message = exception_message(exc)
                with self._retry_lock:
                    self._last_retry_error = message
                if attempt >= self._retry_policy.max_attempts:
                    raise
                with self._retry_lock:
                    self._total_retries += 1
                self._notify_completion()
                delay = self._retry_policy.delay_before(attempt + 1)
                if stop_event.wait(delay):
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
            custom = copy.deepcopy(custom)
            if not isinstance(custom, dict):
                raise TypeError("copied status must be a dictionary")
            snapshot.update(custom)
        return snapshot
