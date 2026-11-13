"""JSON budgets apply before device I/O, retention, and SDK feedback enqueueing."""

import asyncio
import json
import sys
import threading

import pytest

from orchestrion import CommandValidationError, RequestSchema
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
        pytest.param(5e-324, id="positive-subnormal"),
        pytest.param(-5e-324, id="negative-subnormal"),
        pytest.param(sys.float_info.min, id="minimum-normal"),
        pytest.param(sys.float_info.max, id="maximum-finite"),
        pytest.param(-sys.float_info.max, id="negative-maximum-finite"),
        1e-6,
        1e-7,
        1e20,
        1e21,
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


@pytest.mark.parametrize("sign", [-1, 1], ids=["negative", "positive"])
def test_large_numbers_reject_before_decimal_conversion_limits(sign):
    with pytest.raises(PayloadLimitError):
        _snapshot({"value": sign * 10**10000}, max_bytes=100)


@pytest.mark.parametrize(
    "base, value", [(int, 42), (int, -999), (float, -0.0), (float, 1.25)]
)
def test_numeric_subclass_hooks_do_not_affect_json_budget(base, value):
    class Number(base):
        def __repr__(self):
            raise AssertionError("custom numeric repr must not be called")

        __str__ = __repr__

        def __abs__(self):
            raise AssertionError("custom absolute value must not be called")

        def bit_length(self):
            raise AssertionError("custom bit length must not be called")

        def __lt__(self, other):
            raise AssertionError("custom comparison must not be called")

    payload = {"value": [Number(value)]}
    size = len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    assert _snapshot(payload, max_bytes=size) == payload
    with pytest.raises(PayloadLimitError):
        _snapshot(payload, max_bytes=size - 1)


@pytest.mark.parametrize("text", ["plain ASCII", 'quote"', "slash\\", "中😀", "\x00\x7f"])
def test_string_subclass_hooks_do_not_affect_json_budget(text):
    class Text(str):
        def __len__(self):
            raise AssertionError("custom length must not be used")

        def __str__(self):
            raise AssertionError("custom string conversion must not be used")

    payload = {Text("key"): Text(text)}
    size = len(json.dumps({"key": text}, ensure_ascii=True, separators=(",", ":")))
    assert _snapshot(payload, max_bytes=size) == payload
    with pytest.raises(PayloadLimitError):
        _snapshot(payload, max_bytes=size - 1)


def test_string_budget_matches_json_for_every_ascii_character():
    for text in [chr(code) for code in range(128)] + ["\ud800", "\u0080", "\u2028"]:
        size = len(json.dumps(text, ensure_ascii=True))
        assert _snapshot(text, max_bytes=size) == text
        with pytest.raises(PayloadLimitError):
            _snapshot(text, max_bytes=size - 1)


@pytest.mark.parametrize("base", [list, tuple, dict])
@pytest.mark.parametrize("declared_length", [0, 1000])
def test_container_subclass_budget_matches_copied_entries(base, declared_length):
    class Container(base):
        def __len__(self):
            return declared_length

    value = Container({"a": 0, "b": 0}) if base is dict else Container([0, 0])
    expected = _snapshot(value)
    size = len(json.dumps(expected, ensure_ascii=True, separators=(",", ":")))
    assert _snapshot(value, max_bytes=size) == expected
    with pytest.raises(PayloadLimitError):
        _snapshot(value, max_bytes=size - 1)


def test_container_budget_counts_actual_iteration_instead_of_storage():
    class Sequence(list):
        def __iter__(self):
            return iter((0, 0, 0))

    class Mapping(dict):
        def items(self):
            return (("a", 0), ("b", 0), ("c", 0))

    for value in (Sequence([0]), Mapping(a=0)):
        expected = _snapshot(value)
        size = len(json.dumps(expected, separators=(",", ":")))
        assert _snapshot(value, max_bytes=size) == expected
        with pytest.raises(PayloadLimitError):
            _snapshot(value, max_bytes=size - 1)


@pytest.mark.parametrize("max_bytes", [None, 100])
def test_snapshot_validates_keys_from_the_actual_dictionary_items(max_bytes):
    class Mapping(dict):
        def items(self):
            return ((1, "invalid key"),)

    with pytest.raises(TypeError, match="string dictionary keys"):
        _snapshot(Mapping(valid="unused"), max_bytes=max_bytes)


def test_oversized_iterated_container_stops_before_consuming_all_items():
    visited = []

    class Sequence(list):
        def __iter__(self):
            for value in range(1000):
                visited.append(value)
                yield value

    with pytest.raises(PayloadLimitError):
        _snapshot(Sequence(), max_bytes=16)
    assert 0 < len(visited) < 20


def test_output_reference_budget_matches_its_json_preview():
    reference = OutputRef('vision/中"', ("poses", 0, "x"))
    preview = {
        "target": {
            "$output": {"node_id": reference.node_id, "path": list(reference.path)}
        }
    }
    size = len(json.dumps(preview, ensure_ascii=True, separators=(",", ":")))
    snapshot = _snapshot({"target": reference}, references=True, max_bytes=size)
    assert snapshot["target"] is reference
    with pytest.raises(PayloadLimitError):
        _snapshot({"target": reference}, references=True, max_bytes=size - 1)


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


@pytest.mark.parametrize("boundary", ["input", "output"])
@pytest.mark.parametrize(
    "replacement, limit, error",
    [
        ({"value": "x" * 1000}, 32, "PayloadLimitError"),
        ({"value": float("nan")}, None, "TypeError"),
        ({"value": {1, 2}}, None, "TypeError"),
        ([], None, None),
        (None, None, None),
    ],
    ids=["oversized", "nonfinite", "non-json", "array", "none"],
)
def test_schema_returned_payload_is_checked_before_dispatch_or_retention(
    tmp_path, boundary, replacement, limit, error
):
    class ChangedSchema(RequestSchema):
        def validate(self, content):
            super().validate(content)
            return replacement

    async def scenario():
        calls = []

        async def device(context, inputs):
            calls.append(inputs)
            return {"value": 1}

        schema = ChangedSchema({}, allow_extra=True)
        action = Action("device", device, **{boundary + "_schema": schema})
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [action], journal=journal,
                **{"max_" + boundary + "_bytes_per_node": limit},
            ) as engine:
                run = engine.submit(Workflow((Node("device", "device"),)))
                result = await run.wait(1)
                node = result.nodes["device"]
                assert result.status == RunStatus.FAILED
                assert node.status == NodeStatus.FAILED
                expected_error = error or (
                    "CommandValidationError"
                    if boundary == "input" else "ActionOutputError"
                )
                assert node.error.startswith(expected_error + ":")
                assert node.output is None
                assert len(calls) == (1 if boundary == "output" else 0)
                record = journal.get_run(run.run_id)
                assert record["nodes"][0]["output"] is None
                assert not record["nodes"][0]["lease_active"]

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["input", "output"])
def test_schema_copy_cannot_expand_payload_past_its_byte_limit(boundary):
    class ExpandingNumber(int):
        def __deepcopy__(self, memo):
            return "x" * 1000

    async def scenario():
        calls = []
        payload = {"value": ExpandingNumber(1)}

        async def device(context, inputs):
            calls.append(inputs)
            return payload

        schema = RequestSchema({}, allow_extra=True)
        async with WorkflowEngine(
            [Action("device", device, **{boundary + "_schema": schema})],
            **{"max_" + boundary + "_bytes_per_node": 32},
        ) as engine:
            run = engine.submit(Workflow((Node("device", "device", payload),)))
            node = (await run.wait(1)).nodes["device"]
            assert node.status == NodeStatus.FAILED
            assert node.error.startswith("PayloadLimitError:")
            assert node.output is None
            assert len(calls) == (1 if boundary == "output" else 0)

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["input", "output"])
def test_schema_returned_payload_is_isolated_from_external_mutation(boundary):
    shared = {"values": [1]}

    class SharedSchema(RequestSchema):
        def validate(self, content):
            super().validate(content)
            return shared

    async def scenario():
        async def device(context, inputs):
            if boundary == "input":
                inputs["values"].append(2)
            return inputs

        schema = SharedSchema({}, allow_extra=True)
        action = Action("device", device, **{boundary + "_schema": schema})
        async with WorkflowEngine([action]) as engine:
            run = engine.submit(Workflow((Node("device", "device"),)))
            result = await run.wait(1)
            assert result.status == RunStatus.SUCCEEDED
            assert shared == {"values": [1]}
            expected = {"values": [1, 2] if boundary == "input" else [1]}
            assert result.nodes["device"].output == expected
            shared["values"].append(3)
            assert run.snapshot().nodes["device"].output == expected

    asyncio.run(scenario())


def test_oversized_bound_inputs_are_rejected_before_schema_copying():
    schema_calls = []

    class ObservedSchema(RequestSchema):
        def validate(self, content):
            schema_calls.append(content)
            return super().validate(content)

    async def scenario():
        calls = []

        async def source(context, inputs):
            return {"value": "x" * 1000}

        async def consume(context, inputs):
            calls.append(inputs)

        async with WorkflowEngine(
            [
                Action("source", source),
                Action(
                    "consume", consume,
                    input_schema=ObservedSchema({}, allow_extra=True),
                ),
            ],
            max_input_bytes_per_node=128,
        ) as engine:
            workflow = Workflow((
                Node("source", "source"),
                Node(
                    "consume", "consume",
                    {"value": OutputRef("source", ("value",))},
                ),
            ))
            run = engine.submit(workflow)
            result = await run.wait(1)
            assert result.nodes["source"].status == NodeStatus.SUCCEEDED
            assert result.nodes["consume"].error.startswith("PayloadLimitError:")
            assert schema_calls == []
            assert calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "replacement, expected_output",
    [
        (float("nan"), None),
        ({1, 2}, None),
        ("x" * 1000, None),
        (2, {"value": 2}),
    ],
    ids=["nonfinite", "non-json", "oversized", "valid-diagnostic"],
)
@pytest.mark.parametrize("error_type", [CommandValidationError, RuntimeError])
def test_failing_schema_cannot_leave_invalid_output_diagnostics(
    tmp_path, replacement, expected_output, error_type
):
    class RejectingSchema(RequestSchema):
        def validate(self, content):
            content["value"] = replacement
            raise error_type("device output rejected")

    async def scenario():
        async def device(context, inputs):
            return {"value": 1}

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("device", device, output_schema=RejectingSchema({}))],
                journal=journal, max_output_bytes_per_node=32,
            ) as engine:
                run = engine.submit(Workflow((Node("device", "device"),)))
                node = (await run.wait(1)).nodes["device"]
                expected_error = (
                    "ActionOutputError" if error_type is CommandValidationError
                    else "RuntimeError"
                )
                assert node.status == NodeStatus.FAILED
                assert node.error == expected_error + ": device output rejected"
                assert node.output == expected_output
                record = journal.get_run(run.run_id)
                assert record["nodes"][0]["output"] == expected_output

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
            except BaseException as exc:
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
        assert errors == []
        assert context._feedback_snapshot()["dropped"] == 1
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
