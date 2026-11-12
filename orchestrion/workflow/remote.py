"""Transport-independent ticket adapter with explicit device completion semantics."""

import asyncio
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol

from orchestrion.utils.errors import exception_description

from .actions import ActionContext
from .async_tools import (
    AsyncFailure,
    cancel_and_drain,
    drain_start_failure,
    run_rpc,
    start_task,
    task_result,
)
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
        budgets = {}
        for name, value in (
            ("poll_interval", poll_interval),
            ("rpc_timeout", rpc_timeout),
            ("stop_timeout", stop_timeout),
        ):
            value = _timeout(value)
            if value is None:
                raise ValueError(name + " must be positive")
            budgets[name] = value
        _positive_int(max_query_errors, "max_query_errors")
        for method in ("submit", "query", "cancel"):
            if not inspect.iscoroutinefunction(getattr(backend, method, None)):
                raise TypeError("backend.{} must be async".format(method))
        self._backend = backend
        self._poll_interval = budgets["poll_interval"]
        self._rpc_timeout = budgets["rpc_timeout"]
        self._stop_timeout = budgets["stop_timeout"]
        self._max_query_errors = int.__int__(max_query_errors)

    async def _query(self, context: ActionContext, ticket: str, budget: float, on_stop):
        """Interrupt a read when stop is requested, without cancelling submit.

        Drain both helper tasks before returning so that a completed adapter
        cannot leave transport work running outside the engine's accounting.
        A result already available when stop arrives remains a valid observation.
        """
        query = start_task(self._backend.query(ticket))
        if isinstance(query, AsyncFailure):
            return await drain_start_failure(
                query, on_cancel=lambda: context._request_cancel("adapter_interrupted")
            )
        tasks = [query]
        result = None
        try:
            if not context.cancel_requested:
                stopped = start_task(context.wait_cancelled())
                if isinstance(stopped, AsyncFailure):
                    result = await drain_start_failure(stopped)
                else:
                    tasks.append(stopped)
            if result is None:
                done, _ = await asyncio.wait(
                    tasks, timeout=budget, return_when=asyncio.FIRST_COMPLETED
                )
                if context.cancel_requested:
                    # Transport cleanup belongs to the stop-confirmation budget.
                    on_stop()
                if query.done():
                    result = task_result(query)
                elif context.cancel_requested:
                    result = _QUERY_INTERRUPTED
                elif done:
                    result = task_result(tasks[-1])
                else:
                    result = AsyncFailure("TimeoutError: remote query timed out", asyncio.TimeoutError)
        except (Exception, asyncio.CancelledError) as exc:
            result = AsyncFailure(exception_description(exc), type(exc))
            if isinstance(exc, asyncio.CancelledError):
                context._request_cancel("adapter_interrupted")
        finally:
            if await cancel_and_drain(
                tasks, on_cancel=lambda: context._request_cancel("adapter_interrupted")
            ):
                result = AsyncFailure("CancelledError: remote query cleanup interrupted", asyncio.CancelledError)
        if (
            isinstance(result, AsyncFailure)
            and result.error_type is asyncio.TimeoutError
            and not query.cancelled()
        ):
            return task_result(query)
        return result

    async def __call__(self, context: ActionContext, inputs: Dict[str, Any]) -> Any:
        context._check_loop()
        if context.cancel_requested:
            return ActionOutcome.cancelled(context.cancel_reason)
        try:
            ticket = await run_rpc(
                self._backend.submit(context.operation_id, inputs), self._rpc_timeout,
                on_cancel=lambda: context._request_cancel("adapter_interrupted"),
            )
        except OperationRejected as exc:
            return ActionOutcome.failed(exception_description(exc))
        except Exception as exc:
            return ActionOutcome.unknown(
                "remote submission outcome is unknown: " + exception_description(exc)
            )
        if isinstance(ticket, AsyncFailure):
            if ticket.not_started:
                return ActionOutcome.failed("remote submission could not start: " + ticket.error)
            if ticket.error_type is not None and issubclass(ticket.error_type, OperationRejected):
                return ActionOutcome.failed(ticket.error)
            return ActionOutcome.unknown("remote submission outcome is unknown: " + ticket.error)
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
                    cancelled = await run_rpc(
                        self._backend.cancel(ticket), budget,
                        on_cancel=lambda: context._request_cancel("adapter_interrupted"),
                    )
                except Exception:
                    # A lost cancel acknowledgement may still have stopped the
                    # device. Keep querying and retry this idempotent request.
                    pass
                else:
                    if isinstance(cancelled, AsyncFailure):
                        if cancelled.cancelled:
                            return ActionOutcome.unknown("remote cancellation interrupted: " + cancelled.error)
                    else:
                        cancel_accepted = True
            budget = rpc_budget()
            if budget <= 0:
                return ActionOutcome.unknown(
                    "remote stop confirmation deadline expired"
                )
            try:
                update = await self._query(context, ticket, budget, observe_stop)
            except (Exception, asyncio.CancelledError) as exc:
                update = AsyncFailure(exception_description(exc), type(exc))
            if update is _QUERY_INTERRUPTED:
                # Send cancel only after the interrupted read has drained.
                continue
            if stop_deadline is not None and loop.time() >= stop_deadline:
                return ActionOutcome.unknown("remote stop confirmation deadline expired")
            if isinstance(update, AsyncFailure):
                if update.cancelled:
                    return ActionOutcome.unknown("remote query interrupted: " + update.error)
                if update.error_type is not None and issubclass(update.error_type, ExecutionUncertain):
                    return ActionOutcome.unknown(update.error)
                query_errors += 1
                if query_errors >= self._max_query_errors:
                    return ActionOutcome.unknown("remote result is unavailable: " + update.error)
            else:
                if not isinstance(update, OperationUpdate) or update.ticket != ticket:
                    return ActionOutcome.unknown("device returned an invalid or mismatched ticket")
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
                delay = await run_rpc(
                    context.wait_cancelled(), self._poll_interval,
                    on_cancel=lambda: context._request_cancel("adapter_interrupted"),
                )
                if isinstance(delay, AsyncFailure) and (
                    delay.not_started or delay.error_type is not asyncio.TimeoutError
                ):
                    return ActionOutcome.unknown("remote polling wait failed: " + delay.error)
