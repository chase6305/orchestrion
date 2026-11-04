"""JSON budgets apply before device I/O, retention, and SDK feedback enqueueing."""

import asyncio
import json
import threading

import pytest

from orchestrion.workflow import (
    Action,
    ActionContext,
    Node,
    NodeStatus,
    OutputRef,
    PayloadLimitError,
    Resource,
    ResourceClaim,
    RunStatus,
    SQLiteJournal,
    Workflow,
    WorkflowEngine,
)
from orchestrion.workflow.model import _snapshot


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -1,
        10**100,
        -(10**100),
        1.25,
        -0.0,
        "",
        '中😀\n"',
        [],
        {},
        [1, {"x": [True, "hi"]}],
        {"中": {"x": [1, 2, None]}},
    ],
)
def test_snapshot_budget_matches_compact_ascii_json_exactly(value):
    size = len(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    )
    assert _snapshot(value, max_bytes=size) == value
    with pytest.raises(PayloadLimitError):
        _snapshot(value, max_bytes=size - 1)


def test_large_numbers_reject_before_decimal_conversion_limits():
    with pytest.raises(PayloadLimitError):
        _snapshot({"value": 10**10000}, max_bytes=100)


def test_large_literal_is_rejected_before_io_and_does_not_reserve_key(tmp_path):
    calls = []

    async def record(context, inputs):
        calls.append(inputs)
        return inputs

    async def scenario():
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("record", record)],
                journal=journal,
                max_input_bytes_per_node=32,
            ) as engine:
                bad = Workflow((Node("big", "record", {"image": "x" * 1000}),))
                for invoke in (
                    lambda: engine.validate(bad),
                    lambda: engine.submit(bad, idempotency_key="key"),
                ):
                    with pytest.raises(PayloadLimitError):
                        invoke()
                assert engine.get_run_by_key("key") is None
                assert journal._load() == []
                assert calls == []
                good = engine.submit(
                    Workflow((Node("small", "record", {"artifact": "image://42"}),)),
                    idempotency_key="key",
                )
                assert (await good.wait(1)).status == RunStatus.SUCCEEDED
                assert len(calls) == 1

    asyncio.run(scenario())


def test_resolved_inputs_are_checked_before_downstream_device_dispatch():
    async def scenario():
        calls = []

        async def source(context, inputs):
            return {"data": "x" * 200}

        async def consume(context, inputs):
            calls.append(inputs)

        async with WorkflowEngine(
            [
                Action("source", source),
                Action("consume", consume, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")],
            max_input_bytes_per_node=100,
        ) as engine:
            run = engine.submit(
                Workflow(
                    (
                        Node("source", "source"),
                        Node(
                            "consume",
                            "consume",
                            {"data": OutputRef("source", ("data",))},
                        ),
                    )
                )
            )
            result = await run.wait(1)
            assert result.nodes["source"].status == NodeStatus.SUCCEEDED
            assert result.nodes["consume"].status == NodeStatus.FAILED
            assert "PayloadLimitError" in result.nodes["consume"].error
            assert calls == []
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0

    asyncio.run(scenario())


def test_large_output_is_not_retained_journaled_or_given_to_dependents(tmp_path):
    async def scenario():
        async def huge(context, inputs):
            return {"data": "x" * 10000}

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("huge", huge)], journal=journal, max_output_bytes_per_node=32
            ) as engine:
                run = engine.submit(
                    Workflow(
                        (
                            Node("huge", "huge"),
                            Node("after", "huge", after=("huge",)),
                        )
                    )
                )
                result = await run.wait(1)
                assert result.status == RunStatus.FAILED
                assert result.nodes["huge"].output is None
                assert "PayloadLimitError" in result.nodes["huge"].error
                assert result.nodes["after"].status == NodeStatus.SKIPPED
                record = journal.get_run(run.run_id)
                assert record["nodes"][0]["output"] is None
                assert not record["nodes"][0]["lease_active"]

    asyncio.run(scenario())


def test_feedback_budget_is_checked_before_enqueueing_from_sdk_thread():
    async def scenario():
        delivered = []
        errors = []
        finished = asyncio.Event()
        loop = asyncio.get_running_loop()
        context = ActionContext("run", "node", delivered.append, max_feedback_bytes=32)

        def sdk():
            try:
                context.report({"image": "x" * 1000})
            except PayloadLimitError as exc:
                errors.append(str(exc))
            context.report({"progress": 1})
            loop.call_soon_threadsafe(finished.set)

        thread = threading.Thread(target=sdk)
        thread.start()
        try:
            await asyncio.wait_for(finished.wait(), 1)
        finally:
            thread.join(1)
            context._finish()
        assert delivered == [{"progress": 1}]
        assert errors
        assert not context._feedback_snapshot()["pending"]

    asyncio.run(scenario())


def test_budgeted_snapshot_still_rejects_cycles_and_nonfinite_json():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cycles"):
        _snapshot(cycle, max_bytes=100)
    with pytest.raises(TypeError, match="finite JSON"):
        _snapshot({"x": float("nan")}, max_bytes=100)
    with pytest.raises(TypeError, match="string dictionary keys"):
        _snapshot({1: True}, max_bytes=100)


@pytest.mark.parametrize("value", [0, -1, True, "32", 1.5])
@pytest.mark.parametrize(
    "option",
    [
        "max_input_bytes_per_node",
        "max_output_bytes_per_node",
        "max_feedback_bytes_per_report",
    ],
)
def test_invalid_payload_limits_reject_at_engine_construction(option, value):
    with pytest.raises(ValueError):
        WorkflowEngine([], **{option: value})
