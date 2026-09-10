"""Adapter contracts. Completion always means the underlying action is quiescent."""

import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol, Tuple

from orchestrion.schema import RequestSchema

from .feedback import FeedbackChannel
from .model import (
    ActionCancelled,
    ResourceClaim,
    Workflow,
    _name,
    _positive_int,
    _snapshot,
    _timeout,
)


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
    ) -> None:
        _positive_int(max_pending_feedback, "max_pending_feedback")
        self._run_id = run_id
        self._node_id = node_id
        self._loop = asyncio.get_running_loop()
        self._cancelled = asyncio.Event()
        self._cancel_token: Optional[threading.Event] = None
        self._cancel_reason: Optional[str] = None
        self._feedback = feedback
        self._feedback_overflow = feedback_overflow
        self._feedback_limit = max_pending_feedback
        self._feedback_lock = threading.Lock()
        self._feedback_channel: Optional[FeedbackChannel] = None
        self._active = True
        self._handler_started = False

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def operation_id(self) -> str:
        return "{}/{}".format(self._run_id, self._node_id)

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
        _timeout(delay)
        if delay is None:
            raise ValueError("delay must be positive")
        self.raise_if_cancelled()
        try:
            await asyncio.wait_for(self._cancelled.wait(), timeout=delay)
        except asyncio.TimeoutError:
            return
        self.raise_if_cancelled()

    def report(self, feedback: Any) -> None:
        """Report small JSON; SDK-thread overflow drops oldest pending progress."""
        if not self._active:
            return
        snapshot = _snapshot(feedback)
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self._loop:
            if self._active:
                self._feedback(snapshot)
            return
        channel = self._feedback_channel
        if channel is not None:
            channel.post(snapshot)
            return
        with self._feedback_lock:
            if not self._active:
                return
            if self._feedback_channel is None:
                self._feedback_channel = FeedbackChannel(
                    self._loop, self._feedback, self._feedback_overflow,
                    self._feedback_limit,
                )
            channel = self._feedback_channel
        channel.post(snapshot)

    def _check_loop(self) -> None:
        if not self._active:
            raise RuntimeError("ActionContext has already completed")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("ActionContext waits must use their original event loop")

    def _finish(self) -> int:
        with self._feedback_lock:
            self._active = False
            self._feedback = self._feedback_overflow = None
            channel = self._feedback_channel
        return 0 if channel is None else channel.close()

    def _feedback_snapshot(self) -> Dict[str, int]:
        with self._feedback_lock:
            channel = self._feedback_channel
        return {"pending": 0, "dropped": 0} if channel is None else channel.snapshot()

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
        if self._cancel_token is not None:
            self._cancel_token.set()
        self._cancelled.set()
        return True


ActionHandler = Callable[[ActionContext, Dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class Action:
    """A bound capability with mandatory resource claims and an async handler.

    Ordinary exceptions assert the operation has stopped. Network ambiguity must
    use ExecutionUncertain. Do not return an acceptance ticket as a final result.
    """

    name: str
    handler: ActionHandler
    claims: Tuple[ResourceClaim, ...] = ()
    input_schema: Optional[RequestSchema] = None
    output_schema: Optional[RequestSchema] = None

    def __post_init__(self) -> None:
        _name(self.name, "action name")
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
        context.raise_if_cancelled()
        cancellation = context._thread_cancellation()
        future = asyncio.get_running_loop().run_in_executor(
            self._executor, partial(self._invoke, context, cancellation, inputs)
        )
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
                raise ActionCancelled(
                    "executor cancelled the SDK call before start"
                ) from None
            raise

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
            raise ActionCancelled(context.cancel_reason)
        return self._function(inputs)


class Planner(Protocol):
    """A planner proposes a workflow; engine validation still gates execution."""

    async def plan(
        self,
        goal: Mapping[str, Any],
        world: Mapping[str, Any],
        catalog: Mapping[str, Any],
    ) -> Workflow: ...
