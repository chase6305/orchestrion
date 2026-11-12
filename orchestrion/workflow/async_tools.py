"""Owned helper tasks that return failures and finish cleanup before returning."""

import asyncio
import inspect
from dataclasses import dataclass, replace
from typing import Any, Callable, Coroutine, Iterable, Optional, Type, Union

from orchestrion.utils.errors import exception_description
from orchestrion.utils.logger import log_error


@dataclass(frozen=True)
class AsyncFailure:
    """Keep a diagnostic and classification without retaining an SDK traceback."""

    error: str
    error_type: Optional[Type[BaseException]] = None
    not_started: bool = False
    pending_task: Optional[asyncio.Task] = None

    @property
    def cancelled(self) -> bool:
        return self.error_type is not None and issubclass(
            self.error_type, asyncio.CancelledError
        )


def start_task(coroutine: Coroutine[Any, Any, Any]) -> Union[asyncio.Task, AsyncFailure]:
    """Close a rejected coroutine, including when its factory cancels startup."""
    task = None
    pending_task = None
    try:
        task = asyncio.create_task(coroutine)
        return task
    except (Exception, asyncio.CancelledError) as exc:
        # A custom factory may create/start a task and then fail before returning
        # it (for example, its own logging fails). Recover that owner so callers
        # can drain it instead of closing an actively owned coroutine.
        pending_task = next(
            (candidate for candidate in asyncio.all_tasks()
             if candidate.get_coro() is coroutine), None,
        )
        return AsyncFailure(
            exception_description(exc), type(exc),
            not_started=inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED,
            pending_task=pending_task,
        )
    finally:
        if task is None and pending_task is None:
            coroutine.close()


def task_result(task: asyncio.Future) -> Any:
    """Collect a completed task without raising its transport exception."""
    if task.cancelled():
        return AsyncFailure("CancelledError: asynchronous operation cancelled", asyncio.CancelledError)
    error = task.exception()
    if error is not None:
        return AsyncFailure(exception_description(error), type(error))
    return task.result()


def _notify_cancel(callback: Optional[Callable[[], Any]]) -> None:
    if callback is not None:
        try:
            callback()
        except Exception as exc:
            log_error("Helper cancellation observer failed: %s", exception_description(exc))


async def cancel_and_drain(
    tasks: Iterable[asyncio.Future], *, on_cancel: Optional[Callable[[], Any]] = None
) -> bool:
    """Cancel owned work once and drain it even if the caller is cancelled again.

    Return whether caller cancellation interrupted the wait. Task results and
    exceptions are consumed; no transport task outlives its adapter's accounting.
    """
    tasks = tuple(tasks)
    if not tasks:
        return False
    for task in tasks:
        if not task.done():
            task.cancel()
    if all(task.done() for task in tasks):
        for task in tasks:
            if not task.cancelled():
                task.exception()
        return False
    finished = asyncio.gather(*tasks, return_exceptions=True)
    interrupted = False
    while not finished.done():
        try:
            await asyncio.shield(finished)
        except asyncio.CancelledError:
            interrupted = True
            _notify_cancel(on_cancel)
    return interrupted


async def drain_start_failure(
    failure: AsyncFailure, *, on_cancel: Optional[Callable[[], Any]] = None
) -> AsyncFailure:
    """Drain an owner recovered from a failed custom task factory."""
    if failure.pending_task is not None:
        if not failure.not_started:
            _notify_cancel(on_cancel)
        await cancel_and_drain((failure.pending_task,), on_cancel=on_cancel)
        return replace(failure, pending_task=None)
    return failure


async def run_rpc(
    coroutine: Coroutine[Any, Any, Any], timeout: float,
    *, on_cancel: Optional[Callable[[], Any]] = None,
) -> Any:
    """Return an RPC result or AsyncFailure after all owned transport work exits."""
    task = start_task(coroutine)
    if isinstance(task, AsyncFailure):
        return await drain_start_failure(task, on_cancel=on_cancel)
    result = None
    try:
        done, _ = await asyncio.wait((task,), timeout=timeout)
        result = (
            task_result(task) if done else AsyncFailure(
                "TimeoutError: asynchronous operation timed out", asyncio.TimeoutError
            )
        )
    except (Exception, asyncio.CancelledError) as exc:
        result = AsyncFailure(exception_description(exc), type(exc))
        if isinstance(exc, asyncio.CancelledError):
            _notify_cancel(on_cancel)
    finally:
        if await cancel_and_drain((task,), on_cancel=on_cancel):
            result = AsyncFailure("CancelledError: RPC cleanup interrupted", asyncio.CancelledError)
    if (
        isinstance(result, AsyncFailure)
        and result.error_type is asyncio.TimeoutError
        and not task.cancelled()
    ):
        # A transport can finish while acknowledging timeout cancellation. Its
        # accepted ticket or final device observation must not be discarded.
        return task_result(task)
    return result
