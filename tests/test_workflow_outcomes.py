"""Returned adapter outcomes preserve execution, device ownership and recovery."""

import asyncio

import pytest

from orchestrion import FieldSpec, RequestSchema
from orchestrion.utils.logger import logger
from orchestrion.workflow import (
    Action,
    ActionContext,
    ActionOutcome,
    BlockingAction,
    FailurePolicy,
    Node,
    NodeStatus,
    OperationStatus,
    OperationUpdate,
    Resource,
    ResourceClaim,
    RunStatus,
    SQLiteJournal,
    TicketAction,
    Workflow,
    WorkflowEngine,
    workflow_action,
)


@pytest.mark.parametrize(
    "status", [NodeStatus.FAILED, NodeStatus.CANCELLED, NodeStatus.UNKNOWN]
)
@pytest.mark.parametrize("adapter", ["async", "blocking", "scope"])
def test_returned_outcomes_keep_failures_and_device_ownership(status, adapter, caplog):
    async def scenario():
        calls = []
        outcome = ActionOutcome(status, error="controller outcome")

        async def device(context, inputs):
            return outcome

        async def dependent(context, inputs):
            calls.append(context.node_id)

        handler = (
            BlockingAction(lambda inputs: outcome) if adapter == "blocking" else device
        )
        action = Action("device", handler, (ResourceClaim("arm"),))
        if adapter == "scope":
            action = workflow_action(
                "device",
                Workflow((Node("inner", "device"),)),
                [action],
                [Resource("arm")],
            )
        async with WorkflowEngine(
            [action, Action("dependent", dependent)], [Resource("arm")]
        ) as engine:
            run = engine.submit(
                Workflow(
                    (
                        Node("device", "device"),
                        Node("after", "dependent", after=("device",)),
                    )
                )
            )
            result = await run.wait(1)
            assert result.nodes["device"].status == status
            assert result.nodes["after"].status == NodeStatus.SKIPPED
            assert calls == []
            if status == NodeStatus.UNKNOWN:
                assert result.status == RunStatus.NEEDS_ATTENTION
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
                engine.confirm_quiescent(
                    run.operation_id("device"), evidence="controller idle"
                )
            else:
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
            if status != NodeStatus.CANCELLED:
                assert run.operation_id("device") in caplog.text

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "value,expected", [(3, RunStatus.SUCCEEDED), ("bad", RunStatus.FAILED)]
)
def test_explicit_success_still_checks_schema_and_copies_output(value, expected):
    output = {"value": value}

    async def scenario():
        async def device(context, inputs):
            return ActionOutcome(NodeStatus.SUCCEEDED, output)

        schema = RequestSchema({"value": FieldSpec("integer", required=True)})
        async with WorkflowEngine(
            [Action("device", device, output_schema=schema)]
        ) as engine:
            run = engine.submit(Workflow((Node("device", "device"),)))
            result = await run.wait(1)
            assert result.status == expected
            assert result.nodes["device"].output == {"value": value}
            if expected == RunStatus.FAILED:
                assert result.nodes["device"].error.startswith("ActionOutputError:")
            output["value"] = "changed"
            assert run.snapshot().nodes["device"].output == {"value": value}

    asyncio.run(scenario())


class CancelledDiagnostic(dict):
    def items(self):
        raise asyncio.CancelledError("diagnostic capture interrupted")


@pytest.mark.parametrize(
    "invalid",
    [float("nan"), object(), "x" * 100, CancelledDiagnostic()],
    ids=["nonfinite", "non-json", "oversized", "cancelled-snapshot"],
)
@pytest.mark.parametrize(
    "status", [NodeStatus.UNKNOWN, NodeStatus.FAILED, NodeStatus.CANCELLED]
)
def test_invalid_diagnostics_preserve_confirmed_outcomes_and_device_leases(
    tmp_path, invalid, status
):
    async def scenario():
        async def device(context, inputs):
            return ActionOutcome(status, {"value": invalid}, "controller outcome")

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("device", device, (ResourceClaim("arm"),))],
                [Resource("arm")],
                journal=journal,
                max_output_bytes_per_node=32,
            ) as engine:
                run = engine.submit(Workflow((Node("device", "device"),)))
                result = await run.wait(1)
                assert (
                    result.status
                    is {
                        NodeStatus.UNKNOWN: RunStatus.NEEDS_ATTENTION,
                        NodeStatus.FAILED: RunStatus.FAILED,
                        NodeStatus.CANCELLED: RunStatus.CANCELLED,
                    }[status]
                )
                assert result.nodes["device"].status is status
                assert result.nodes["device"].output is None
                assert result.nodes["device"].error == "controller outcome"
                stored = journal.get_run(run.run_id)["nodes"][0]
                assert stored["status"] == status.value
                assert stored["lease_active"] == (status is NodeStatus.UNKNOWN)
                assert stored["output"] is None
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == int(
                    status is NodeStatus.UNKNOWN
                )
                if status is NodeStatus.UNKNOWN:
                    engine.confirm_quiescent(
                        run.operation_id("device"), evidence="controller idle"
                    )

    asyncio.run(scenario())


def test_logging_failure_cannot_interrupt_workflow_accounting(monkeypatch):
    def broken_log(*args, **kwargs):
        raise RuntimeError("log sink unavailable")

    monkeypatch.setattr(logger, "error", broken_log)

    async def scenario():
        async def device(context, inputs):
            return ActionOutcome.failed("device offline")

        async def echo(context, inputs):
            return inputs

        async with WorkflowEngine(
            [Action("device", device), Action("echo", echo)]
        ) as engine:
            run = engine.submit(Workflow((Node("device", "device"),)))
            assert (await run.wait(1)).nodes["device"].error == "device offline"
            run = engine.submit(Workflow((Node("echo", "echo", {"ok": True}),)))
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED
            assert engine.scheduler_snapshot()["running_actions"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status",
    [OperationStatus.FAILED, OperationStatus.CANCELLED, OperationStatus.UNKNOWN],
)
def test_ticket_adapter_returns_expected_terminal_outcomes_directly(status):
    async def scenario():
        class Backend:
            async def submit(self, operation_id, inputs):
                return "ticket"

            async def query(self, ticket):
                return OperationUpdate(ticket, status, error="controller outcome")

            async def cancel(self, ticket):
                pass

        context = ActionContext("run", "node", lambda data: None)
        result = await TicketAction(Backend())(context, {})
        assert isinstance(result, ActionOutcome)
        assert result.status.value == status.value
        assert result.error == "controller outcome"

    asyncio.run(scenario())


def test_returned_failure_preserves_independent_work():
    async def scenario():
        async def device(context, inputs):
            return ActionOutcome.failed("rejected")

        async def other(context, inputs):
            return {"ok": True}

        async with WorkflowEngine(
            [Action("device", device), Action("other", other)]
        ) as engine:
            run = engine.submit(
                Workflow(
                    (Node("device", "device"), Node("other", "other")),
                    FailurePolicy.CONTINUE_INDEPENDENT,
                )
            )
            result = await run.wait(1)
            assert result.status == RunStatus.FAILED
            assert result.nodes["other"].status == NodeStatus.SUCCEEDED
            assert result.nodes["other"].output == {"ok": True}

    asyncio.run(scenario())
