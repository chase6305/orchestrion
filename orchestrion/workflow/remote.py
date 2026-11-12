"""Transport-independent ticket adapter with explicit device completion semantics."""

import asyncio
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol

from orchestrion.utils.errors import exception_description

from .actions import ActionContext
from .model import (
    ActionOutcome,
    ExecutionUncertain,
    _name,
    _positive_int,
    _snapshot,
    _timeout,
)

_QUERY_INTERRUPTED = object()


class OperationStatus(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OperationUpdate:
    """A remote observation; FAILED/CANCELLED must prove the operation has stopped."""

    ticket: str
    status: OperationStatus
    output: Any = None
    error: Optional[str] = None
    feedback: Any = None

    def __post_init__(self) -> None:
        _name(self.ticket, "operation ticket")
        object.__setattr__(self, "status", OperationStatus(self.status))
        object.__setattr__(self, "output", _snapshot(self.output))
        object.__setattr__(self, "feedback", _snapshot(self.feedback))
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("operation error must be a string or None")


class OperationRejected(RuntimeError):
    """Submit was definitely refused; the device cannot execute this operation."""


class RemoteOperationError(RuntimeError):
    """A remote operation failed and its controller confirms it has stopped."""


class OperationBackend(Protocol):
    """Async device transport; configure bounded I/O and cooperative coroutine exit.

    submit must pass operation_id as the protocol's deduplication key. A submit
    retry must return the original ticket and must never repeat an actuator write.
    Raise OperationRejected only for definite refusal. Cancellation acceptance is
    not a stop proof; query must return a final state after controller confirmation.
    """

    async def submit(self, operation_id: str, inputs: Dict[str, Any]) -> str: ...

    async def query(self, ticket: str) -> OperationUpdate: ...

    async def cancel(self, ticket: str) -> None: ...


class TicketAction:
    """Wait for remote completion, including after a cooperative stop request.

    Bind with Action(name, TicketAction(backend), claims=...). No actuator submit
    retries are performed here. Query transport errors are bounded; cancellation
    requests may be repeated and must be idempotent. Lost submissions, exhausted
    queries, bad tickets and stop timeouts return UNKNOWN outcomes. Optional
    malformed/oversized progress reports cannot release an active device lease.
    """

    def __init__(
        self,
        backend: OperationBackend,
        *,
        poll_interval: float = 0.05,
        rpc_timeout: float = 5.0,
        stop_timeout: float = 5.0,
        max_query_errors: int = 3,
    ) -> None:
        for name, value in (
            ("poll_interval", poll_interval),
            ("rpc_timeout", rpc_timeout),
            ("stop_timeout", stop_timeout),
        ):
            _timeout(value)
            if value is None:
                raise ValueError(name + " must be positive")
        _positive_int(max_query_errors, "max_query_errors")
        for method in ("submit", "query", "cancel"):
            if not inspect.iscoroutinefunction(getattr(backend, method, None)):
                raise TypeError("backend.{} must be async".format(method))
        self._backend = backend
        self._poll_interval = poll_interval
        self._rpc_timeout = rpc_timeout
        self._stop_timeout = stop_timeout
        self._max_query_errors = max_query_errors

    async def _query(self, context: ActionContext, ticket: str, budget: float, on_stop):
        """Interrupt a read when stop is requested, without cancelling submit.

        Drain both helper tasks before returning so that a completed adapter
        cannot leave transport work running outside the engine's accounting.
        A result already available when stop arrives remains a valid observation.
        """
        if context.cancel_requested:
            return await asyncio.wait_for(self._backend.query(ticket), budget)
        query = asyncio.ensure_future(
            asyncio.wait_for(self._backend.query(ticket), budget)
        )
        stopped = asyncio.ensure_future(context.wait_cancelled())
        try:
            await asyncio.wait((query, stopped), return_when=asyncio.FIRST_COMPLETED)
            if context.cancel_requested:
                # Include transport cleanup in the confirmation budget. Starting
                # the clock only after draining would hide a slow read shutdown.
                on_stop()
            if query.done():
                return query.result()
            return _QUERY_INTERRUPTED
        finally:
            for task in (query, stopped):
                if not task.done():
                    task.cancel()
            await asyncio.gather(query, stopped, return_exceptions=True)

    async def __call__(self, context: ActionContext, inputs: Dict[str, Any]) -> Any:
        context._check_loop()
        if context.cancel_requested:
            return ActionOutcome.cancelled(context.cancel_reason)
        try:
            ticket = await asyncio.wait_for(
                self._backend.submit(context.operation_id, inputs), self._rpc_timeout
            )
        except OperationRejected as exc:
            return ActionOutcome.failed(exception_description(exc))
        except Exception as exc:
            return ActionOutcome.unknown(
                "remote submission outcome is unknown: " + exception_description(exc)
            )
        if not isinstance(ticket, str) or not ticket.strip():
            return ActionOutcome.unknown("device returned an invalid operation ticket")

        try:
            context.checkpoint({"ticket": ticket, "operation_id": context.operation_id})
        except Exception as exc:
            return ActionOutcome.unknown(
                "accepted remote ticket could not be recorded: "
                + exception_description(exc)
            )

        loop = asyncio.get_running_loop()
        stop_deadline = None
        cancel_accepted = False
        query_errors = 0

        def observe_stop():
            nonlocal stop_deadline
            if context.cancel_requested and stop_deadline is None:
                stop_deadline = loop.time() + self._stop_timeout

        def rpc_budget():
            if stop_deadline is None:
                return self._rpc_timeout
            remaining = stop_deadline - loop.time()
            return max(0.0, min(self._rpc_timeout, remaining))

        while True:
            observe_stop()
            if stop_deadline is not None and not cancel_accepted:
                budget = rpc_budget()
                if budget <= 0:
                    return ActionOutcome.unknown(
                        "remote stop confirmation deadline expired"
                    )
                try:
                    await asyncio.wait_for(self._backend.cancel(ticket), budget)
                except Exception:
                    # A lost cancel acknowledgement may still have stopped the
                    # device. Keep querying and retry this idempotent request.
                    pass
                else:
                    cancel_accepted = True
            budget = rpc_budget()
            if budget <= 0:
                return ActionOutcome.unknown(
                    "remote stop confirmation deadline expired"
                )
            try:
                update = await self._query(context, ticket, budget, observe_stop)
                if update is _QUERY_INTERRUPTED:
                    # A stop request interrupted the read. Send cancel on the
                    # next iteration rather than waiting out the normal RPC.
                    continue
                if stop_deadline is not None and loop.time() >= stop_deadline:
                    return ActionOutcome.unknown(
                        "remote stop confirmation deadline expired"
                    )
                if not isinstance(update, OperationUpdate) or update.ticket != ticket:
                    return ActionOutcome.unknown(
                        "device returned an invalid or mismatched ticket"
                    )
            except ExecutionUncertain as exc:
                return ActionOutcome.unknown(exception_description(exc))
            except Exception as exc:
                query_errors += 1
                if query_errors >= self._max_query_errors:
                    return ActionOutcome.unknown(
                        "remote result is unavailable: " + exception_description(exc)
                    )
            else:
                query_errors = 0
                if update.status == OperationStatus.SUCCEEDED:
                    return update.output
                if update.status == OperationStatus.CANCELLED:
                    return ActionOutcome.cancelled(
                        update.error or "remote operation stopped"
                    )
                if update.status == OperationStatus.FAILED:
                    return ActionOutcome.failed(
                        update.error or "remote operation failed"
                    )
                if update.status == OperationStatus.UNKNOWN:
                    return ActionOutcome.unknown(
                        update.error or "remote outcome is unknown"
                    )
                if update.feedback is not None:
                    try:
                        context.report(update.feedback)
                    except Exception:
                        # Progress is optional; final output carries required data.
                        pass
            if context.cancel_requested:
                if stop_deadline is None:
                    continue
                await asyncio.sleep(min(self._poll_interval, rpc_budget()))
            else:
                try:
                    await asyncio.wait_for(
                        context.wait_cancelled(), self._poll_interval
                    )
                except asyncio.TimeoutError:
                    pass
