"""Compose a fixed workflow into one capability with a continuous resource lease."""

import asyncio
import hashlib
import json
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, Optional

from orchestrion.schema import RequestSchema

from .actions import Action, ActionContext
from .compiler import compile_workflow, submission_fingerprint
from .engine import WorkflowEngine
from .model import (
    ActionCancelled,
    ExecutionUncertain,
    Resource,
    ResourceClaim,
    RunStatus,
    Workflow,
    _operation_id,
    _snapshot,
)


def _report_optional(context: ActionContext, feedback: Any) -> None:
    try:
        context.report(feedback)
    except Exception:
        # Mirrored progress cannot decide whether a device has stopped.
        pass


class WorkflowActionError(RuntimeError):
    """A scoped workflow failed after all of its known operations stopped."""


class _ScopedContext(ActionContext):
    """Delegate cancellation while giving drivers stable hierarchical identities."""

    def __init__(
        self,
        delegate: ActionContext,
        parent: ActionContext,
        checkpoint: Callable[[str, Any], None],
    ) -> None:
        self._delegate = delegate
        self._parent = parent
        self._scope_checkpoint = checkpoint
        self._scoped_operation_id = _operation_id(parent.operation_id, delegate.node_id)

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    @property
    def run_id(self) -> str:
        return self._parent.operation_id

    @property
    def node_id(self) -> str:
        return self._delegate.node_id

    @property
    def operation_id(self) -> str:
        return self._scoped_operation_id

    @property
    def cancel_requested(self) -> bool:
        return self._delegate.cancel_requested

    @property
    def cancel_reason(self) -> Optional[str]:
        return self._delegate.cancel_reason

    async def wait_cancelled(self) -> str:
        return await self._delegate.wait_cancelled()

    async def sleep(self, delay: float) -> None:
        await self._delegate.sleep(delay)

    def raise_if_cancelled(self) -> None:
        self._delegate.raise_if_cancelled()

    def report(self, feedback: Any) -> None:
        self._delegate.report(feedback)
        _report_optional(
            self._parent, {"scope_node": self.node_id, "feedback": feedback}
        )

    def _thread_cancellation(self):
        return self._delegate._thread_cancellation()

    def checkpoint(self, data: Any) -> None:
        self._delegate.checkpoint(data)
        self._scope_checkpoint(self.node_id, data)


def workflow_action(
    name: str,
    workflow: Workflow,
    actions: Iterable[Action],
    resources: Iterable[Resource] = (),
    *,
    max_concurrency: int = 32,
    stop_timeout: Optional[float] = None,
    max_input_bytes_per_node: Optional[int] = None,
    max_output_bytes_per_node: Optional[int] = None,
    max_feedback_bytes_per_report: Optional[int] = None,
) -> Action:
    """Expose a fixed DAG as one capability with a continuous resource scope.

    Reserve every supplied Resource at its full declared capacity before entering
    any inner node. Internal nodes share these capacities locally. This avoids
    gaps between steps and nested resource acquisition deadlocks. An uncertain
    inner operation quarantines the whole scope. Inputs are empty; bind inputs in
    the fixed Workflow. The output's ``nodes`` maps step IDs to status/output/error.
    Drivers receive stable IDs ``outer_run/outer_node/inner_node``. The outer engine
    journals the scope as one operation and never replays inner work after a crash.
    """
    handler = _WorkflowAction(
        workflow,
        actions,
        resources,
        {
            "max_concurrency": max_concurrency,
            "stop_timeout": stop_timeout,
            "max_input_bytes_per_node": max_input_bytes_per_node,
            "max_output_bytes_per_node": max_output_bytes_per_node,
            "max_feedback_bytes_per_report": max_feedback_bytes_per_report,
        },
    )
    return Action(
        name,
        handler,
        tuple(ResourceClaim(item.name, item.capacity) for item in handler.resources),
        input_schema=RequestSchema({}),
        version=handler.version,
    )


class _WorkflowAction:
    def __init__(
        self,
        workflow: Workflow,
        actions: Iterable[Action],
        resources: Iterable[Resource],
        options: Dict[str, Any],
    ) -> None:
        # Reuse registration validation and defensive schema snapshots, without
        # creating an event loop, device connections, or any worker tasks.
        registry = WorkflowEngine(actions, resources, **options)
        compiled = compile_workflow(
            workflow,
            registry._actions,
            registry._resources,
            registry._max_nodes,
            registry._max_input_bytes,
        )
        snapshot = replace(workflow, nodes=tuple(item.node for item in compiled))
        self._workflow = snapshot
        self._actions = tuple(registry._actions.values())
        self.resources = tuple(registry._resources.values())
        self._options = options
        definition = {
            "workflow": submission_fingerprint(snapshot, compiled),
            "options": options,
        }
        self.version = (
            "workflow:"
            + hashlib.sha256(
                json.dumps(definition, sort_keys=True, allow_nan=False).encode("ascii")
            ).hexdigest()
        )

    async def __call__(self, context: ActionContext, inputs: Dict[str, Any]) -> Any:
        if inputs:
            raise ValueError("workflow_action uses a fixed plan and accepts no inputs")
        context.raise_if_cancelled()
        checkpoints = {}

        def checkpoint(node_id, data):
            checkpoints[node_id] = _snapshot(data)
            context.checkpoint({"scope_nodes": checkpoints})

        def wrap(action):
            async def invoke(inner_context, inner_inputs):
                _report_optional(
                    context, {"scope_node": inner_context.node_id, "kind": "started"}
                )
                return await action.handler(
                    _ScopedContext(inner_context, context, checkpoint), inner_inputs
                )

            return replace(action, handler=invoke)

        engine = WorkflowEngine(
            [wrap(action) for action in self._actions],
            self.resources,
            **self._options,
        )
        try:
            async with engine:
                run = engine.submit(self._workflow)

                async def forward_cancel():
                    await context.wait_cancelled()
                    run.cancel()

                cancellation = asyncio.create_task(forward_cancel())
                try:
                    result = await run.wait()
                except BaseException:
                    # Cancelling the wrapper cannot silently orphan its inner graph.
                    run.cancel()
                    raise
                finally:
                    cancellation.cancel()
                    await asyncio.gather(cancellation, return_exceptions=True)
            # wait() already isolated these outputs; the outer engine snapshots
            # the returned detail before retaining or binding it to later nodes.
            detail = {
                "nodes": {
                    node_id: {
                        "status": node.status.value,
                        "output": node.output,
                        "error": node.error,
                    }
                    for node_id, node in result.nodes.items()
                }
            }
            _report_optional(
                context,
                {
                    "scope_result": {
                        "nodes": {
                            node_id: {"status": node["status"], "error": node["error"]}
                            for node_id, node in detail["nodes"].items()
                        }
                    }
                },
            )
            if engine.resource_snapshot()["quarantined"]:
                raise ExecutionUncertain(
                    "scoped workflow contains unresolved device operations"
                )
            if result.status == RunStatus.CANCELLED:
                raise ActionCancelled("scoped workflow stopped")
            if result.status != RunStatus.SUCCEEDED:
                errors = "; ".join(
                    "{}: {}".format(node_id, node.error)
                    for node_id, node in result.nodes.items()
                    if node.error
                )
                raise WorkflowActionError(errors or "scoped workflow failed")
            return detail
        except ExecutionUncertain:
            raise
        except Exception as exc:
            if engine._quarantined or engine._workers or engine._runs:
                raise ExecutionUncertain(
                    "scoped workflow could not confirm all operations stopped"
                ) from exc
            raise
