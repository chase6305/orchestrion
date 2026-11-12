"""Adapter contracts. Completion always means the underlying action is quiescent."""

import asyncio
import inspect
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
)

from orchestrion.schema import RequestSchema
from orchestrion.utils.errors import exception_description
from orchestrion.utils.logger import log_error

from .feedback import FeedbackChannel, deliver_feedback
from .model import (
    ActionCancelled,
    ActionOutcome,
    ExecutionUncertain,
    ResourceClaim,
    Workflow,
    _name,
    _operation_id,
    _positive_int,
    _snapshot,
    _timeout,
)

_cancellation_dispatch = threading.local()


def _dispatch_cancellation(callbacks: Iterable[Callable[[], Any]]) -> None:
    """Relay nested scope cancellation without recursive task/stack growth."""
    pending = getattr(_cancellation_dispatch, "pending", None)
    if pending is not None:
        pending.extend(callbacks)
        return
    pending = deque(callbacks)
    _cancellation_dispatch.pending = pending
    try:
        while pending:
            callback = pending.popleft()
            try:
                callback()
            except (Exception, asyncio.CancelledError) as exc:
                log_error("Scope cancellation callback failed: %s", exception_description(exc))
    finally:
        _cancellation_dispatch.pending = None


class ActionContext:
    """Loop-local cancellation waits and thread-safe feedback for one invocation.

    An adapter must finish cleanup before returning/raising ActionCancelled.
    Raise ExecutionUncertain when remote termination cannot be established.
    """

    def __init__(
        self,
        run_id: str,
        node_id: str,
        feedback: Callable[[Any], None],
        *,
        max_pending_feedback: int = 128,
        feedback_overflow: Optional[Callable[[int], None]] = None,
        max_feedback_bytes: Optional[int] = None,
        checkpoint: Optional[Callable[[Any], None]] = None,
    ) -> None:
        _positive_int(max_pending_feedback, "max_pending_feedback")
        self._run_id = run_id
        self._node_id = node_id
        self._operation_id = _operation_id(run_id, node_id)
        self._loop = asyncio.get_running_loop()
        self._cancelled = asyncio.Event()
        self._cancel_token: Optional[threading.Event] = None
        self._cancel_reason: Optional[str] = None
        self._feedback = feedback
        self._feedback_overflow = feedback_overflow
        self._feedback_limit = int.__int__(max_pending_feedback)
        if max_feedback_bytes is not None:
            _positive_int(max_feedback_bytes, "max_feedback_bytes")
        self._max_feedback_bytes = (
            None if max_feedback_bytes is None else int.__int__(max_feedback_bytes)
        )
        self._checkpoint = checkpoint
        self._late_checkpoint = False
        self._feedback_lock = threading.Lock()
        self._feedback_channel: Optional[FeedbackChannel] = None
        self._direct_feedback_dropped = 0
        self._active = True
        self._handler_started = False
        self._stop_deadline: Optional[float] = None
        self._cancel_observer: Optional[Callable[[str], None]] = None
        self._cancel_callbacks: Optional[List[Callable[[], Any]]] = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def operation_id(self) -> str:
        return self._operation_id

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_reason is not None

    @property
    def cancel_reason(self) -> Optional[str]:
        return self._cancel_reason

    async def wait_cancelled(self) -> str:
        self._check_loop()
        await self._cancelled.wait()
        return self._cancel_reason

    def raise_if_cancelled(self) -> None:
        if self.cancel_requested:
            raise ActionCancelled(self._cancel_reason)

    async def sleep(self, delay: float) -> None:
        """An interruptible positive delay, useful for cooperative adapters."""
        self._check_loop()
        delay = _timeout(delay)
        if delay is None:
            raise ValueError("delay must be positive")
        self.raise_if_cancelled()
        try:
            await asyncio.wait_for(self._cancelled.wait(), timeout=delay)
        except asyncio.TimeoutError:
            return
        self.raise_if_cancelled()

    def report(self, feedback: Any) -> None:
        """Report JSON; invalid or undeliverable optional progress logs and drops."""
        self._report_feedback(feedback)

    def _report_feedback(self, feedback: Any) -> Any:
        """Report once and return captured JSON for enclosing scopes."""
        if not self._active:
            return
        try:
            snapshot = _snapshot(feedback, max_bytes=self._max_feedback_bytes)
        except (Exception, asyncio.CancelledError) as exc:
            with self._feedback_lock:
                if not self._active:
                    return
                self._direct_feedback_dropped += 1
            log_error("Workflow feedback was rejected: %s", exception_description(exc))
            return
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self._loop:
            if self._active:
                if not deliver_feedback(self._feedback, snapshot):
                    with self._feedback_lock:
                        self._direct_feedback_dropped += 1
            return snapshot
        channel = self._feedback_channel
        if channel is not None:
            channel.post(snapshot)
            return snapshot
        with self._feedback_lock:
            if not self._active:
                return
            if self._feedback_channel is None:
                self._feedback_channel = FeedbackChannel(
                    self._loop,
                    self._feedback,
                    self._feedback_overflow,
                    self._feedback_limit,
                )
            channel = self._feedback_channel
        channel.post(snapshot)
        return snapshot

    def checkpoint(self, data: Any) -> None:
        """Record small recovery metadata on the event loop before continuing I/O.

        This replaces the previous checkpoint (64 KiB compact ASCII JSON limit).
        With a journal, return means the metadata is durably committed. A failed
        commit raises; an adapter with an accepted operation must report UNKNOWN.
        Unlike optional progress, recovery metadata must never be silently lost.
        A still-draining adapter may record a late ticket after a stop deadline;
        this does not change UNKNOWN or release its resources.
        """
        self._record_checkpoint(data)

    def _record_checkpoint(self, data: Any) -> Any:
        """Record once and return the owned snapshot for enclosing scopes."""
        if not self._active and not self._late_checkpoint:
            raise RuntimeError("ActionContext has already completed")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError(
                "ActionContext checkpoints must use their original event loop"
            )
        snapshot = _snapshot(data, max_bytes=65536)
        if self._checkpoint is not None:
            self._checkpoint(snapshot)
        return snapshot

    def _check_loop(self) -> None:
        if not self._active:
            raise RuntimeError("ActionContext has already completed")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("ActionContext waits must use their original event loop")

    def _finish(self, *, draining: bool = False) -> int:
        with self._feedback_lock:
            retired = (self._feedback, self._feedback_overflow, self._cancel_observer, self._cancel_callbacks)
            self._active = False
            self._feedback = self._feedback_overflow = None
            self._cancel_observer = None
            self._cancel_callbacks = None
            self._late_checkpoint = draining
            if not draining:
                retired += (self._checkpoint,)
                self._checkpoint = None
            channel = self._feedback_channel
            direct_drops = self._direct_feedback_dropped
        del retired
        return direct_drops + (0 if channel is None else channel.close())

    def _feedback_snapshot(self) -> Dict[str, int]:
        with self._feedback_lock:
            channel = self._feedback_channel
            direct_drops = self._direct_feedback_dropped
        snapshot = {"pending": 0, "dropped": 0} if channel is None else channel.snapshot()
        snapshot["dropped"] += direct_drops
        return snapshot

    def _thread_cancellation(self) -> threading.Event:
        self._check_loop()
        if self._cancel_token is None:
            self._cancel_token = threading.Event()
            if self.cancel_requested:
                self._cancel_token.set()
        return self._cancel_token

    def _request_cancel(self, reason: str) -> bool:
        if self.cancel_requested:
            return False
        self._cancel_reason = reason
        callbacks, self._cancel_callbacks = self._cancel_callbacks, None
        try:
            if self._cancel_token is not None:
                self._cancel_token.set()
            self._cancelled.set()
            if self._cancel_observer is not None:
                self._cancel_observer(reason)
        finally:
            if callbacks:
                _dispatch_cancellation(callbacks)
        return True

    def _add_cancel_callback(self, callback: Callable[[], Any]) -> None:
        if self.cancel_requested:
            _dispatch_cancellation((callback,))
            return
        self._check_loop()
        if self._cancel_callbacks is None:
            self._cancel_callbacks = []
        self._cancel_callbacks.append(callback)

    def _remove_cancel_callback(self, callback: Callable[[], Any]) -> None:
        callbacks = self._cancel_callbacks
        if callbacks is not None and callback in callbacks:
            callbacks.remove(callback)
            if not callbacks:
                self._cancel_callbacks = None


ActionHandler = Callable[[ActionContext, Dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class Action:
    """A bound capability with mandatory resource claims and an async handler.

    Return ActionOutcome for explicit failure, cancellation or uncertain device
    state; the engine logs failures and quarantines UNKNOWN resources. Legacy
    exceptions are still contained at the execution boundary. Do not return an
    acceptance ticket as a final result.
    """

    name: str
    handler: ActionHandler
    claims: Tuple[ResourceClaim, ...] = ()
    input_schema: Optional[RequestSchema] = None
    output_schema: Optional[RequestSchema] = None
    version: Optional[str] = None

    def __post_init__(self) -> None:
        _name(self.name, "action name")
        if self.version is not None:
            _name(self.version, "action version")
        if not callable(self.handler) or not (
            inspect.iscoroutinefunction(self.handler)
            or inspect.iscoroutinefunction(getattr(self.handler, "__call__", None))
        ):
            raise TypeError(
                "action handler must be async; wrap SDK calls in BlockingAction"
            )
        object.__setattr__(self, "claims", tuple(self.claims))
        if any(not isinstance(claim, ResourceClaim) for claim in self.claims):
            raise TypeError("action claims must contain ResourceClaim objects")
        if self.input_schema is not None and not isinstance(
            self.input_schema, RequestSchema
        ):
            raise TypeError("input_schema must be RequestSchema or None")
        if self.output_schema is not None and not isinstance(
            self.output_schema, RequestSchema
        ):
            raise TypeError("output_schema must be RequestSchema or None")


class BlockingAction:
    """Run a blocking SDK operation off the event loop, without early lease release.

    The function must wait for physical completion, including on errors. A cancel
    request waits for the function to finish. The caller owns a supplied executor.
    """

    def __init__(
        self,
        function: Callable[[Dict[str, Any]], Any],
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        if not callable(function) or (
            inspect.iscoroutinefunction(function)
            or inspect.iscoroutinefunction(getattr(function, "__call__", None))
        ):
            raise TypeError("BlockingAction requires a synchronous callable")
        if executor is not None and not isinstance(executor, ThreadPoolExecutor):
            raise TypeError("BlockingAction requires a ThreadPoolExecutor or None")
        self._function = function
        self._executor = executor

    async def __call__(self, context: ActionContext, inputs: Dict[str, Any]) -> Any:
        cancellation = context._thread_cancellation()
        if cancellation.is_set():
            return ActionOutcome.cancelled(context.cancel_reason)
        try:
            future = asyncio.get_running_loop().run_in_executor(
                self._executor, partial(self._invoke, context, cancellation, inputs)
            )
        except Exception as exc:
            return ActionOutcome.failed(exception_description(exc))
        future.add_done_callback(self._observe_completion)
        # Unexpected coroutine cancellation becomes UNKNOWN in the engine; the
        # worker is not cancelled and its device resources remain quarantined.
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if future.cancelled():
                # Executor shutdown can cancel a queued callable before it starts.
                # Shielding keeps cancellation of this adapter from marking the
                # underlying future cancelled, so this is a quiescence proof.
                return ActionOutcome.cancelled(
                    "executor cancelled the SDK call before start"
                )
            context._request_cancel("adapter_interrupted")
            # Keep the SDK wait in engine accounting until the worker exits.
            # Repeated cancellation must not detach a still-running device call.
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    continue
            return ActionOutcome.unknown(
                "CancelledError: blocking action wait interrupted"
            )

    @staticmethod
    def _observe_completion(future: asyncio.Future) -> None:
        # shield() detaches its observer if the awaiting coroutine is cancelled.
        # Still collect a late SDK exception; the engine has recorded UNKNOWN and
        # retains the device lease until independent quiescence confirmation.
        if not future.cancelled():
            future.exception()

    def _invoke(
        self,
        context: ActionContext,
        cancellation: threading.Event,
        inputs: Dict[str, Any],
    ) -> Any:
        # A submitted wrapper can wait in an executor queue for a long time. The
        # thread-safe token prevents a known cancellation from becoming device I/O
        # when that wrapper eventually gets a worker. Already started calls drain.
        if cancellation.is_set():
            return ActionOutcome.cancelled(context.cancel_reason)
        try:
            return self._function(inputs)
        except ActionCancelled as exc:
            return ActionOutcome.cancelled(exception_description(exc))
        except ExecutionUncertain as exc:
            return ActionOutcome.unknown(exception_description(exc))
        except Exception as exc:
            return ActionOutcome.failed(exception_description(exc))


class Planner(Protocol):
    """A planner proposes a workflow; engine validation still gates execution."""

    async def plan(
        self,
        goal: Mapping[str, Any],
        world: Mapping[str, Any],
        catalog: Mapping[str, Any],
    ) -> Workflow: ...
