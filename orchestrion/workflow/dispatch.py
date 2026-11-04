"""One event-loop-turn barrier for durable, atomic dispatch intent commits."""

from typing import TYPE_CHECKING

from .journal import JournalError
from .model import NodeResult

if TYPE_CHECKING:
    from .engine import WorkflowEngine
    from .run import WorkflowRun


class DispatchBarrier:
    """Combine concurrent intents without allowing an uncommitted device effect.

    Cancellation before the flush omits the intent. Cancellation or an elapsed
    deadline after the flush is checked again by the worker before adapter entry.
    The queue is bounded by the engine's occupied concurrency slots.
    """

    def __init__(self, engine: "WorkflowEngine") -> None:
        self._engine = engine
        self._pending = []
        self._scheduled = False

    async def commit(self, run: "WorkflowRun", result: NodeResult) -> None:
        future = self._engine._loop.create_future()
        entry = (run, result, future)
        self._pending.append(entry)
        if not self._scheduled:
            try:
                self._engine._loop.call_soon(self._flush)
            except BaseException:
                self._pending.remove(entry)
                future.cancel()
                raise
            self._scheduled = True
        await future

    def _flush(self) -> None:
        pending, self._pending = self._pending, []
        self._scheduled = False
        entries = []
        for run, result, future in pending:
            if future.done():
                continue
            context = run._contexts.get(result.node_id)
            if (
                context is None
                or context.cancel_requested
                or run._results[result.node_id].status.terminal
            ):
                future.set_result(None)
            else:
                entries.append((run, result, future))
        if not entries:
            return
        engine = self._engine
        try:
            if len(entries) == 1:
                run, result, _ = entries[0]
                engine._record_journal("_start", run.run_id, result, strict=True)
            else:
                engine._record_journal(
                    "_start_many",
                    tuple((run.run_id, result) for run, result, _ in entries),
                    strict=True,
                )
        except JournalError as exc:
            for _, _, future in entries:
                future.set_exception(JournalError(str(exc)))
        else:
            for _, _, future in entries:
                future.set_result(None)
