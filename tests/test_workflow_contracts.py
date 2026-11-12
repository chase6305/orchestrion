"""Workflow definitions, input/output contracts, and downstream data binding."""

import asyncio
import threading

import pytest

from orchestrion import CommandValidationError, FieldSpec, RequestSchema
from orchestrion.workflow import (
    Action,
    ActionOutcome,
    BlockingAction,
    ExecutionUncertain,
    FailurePolicy,
    Node,
    NodeStatus,
    OutputRef,
    Resource,
    ResourceClaim,
    RunStatus,
    Workflow,
    WorkflowEngine,
)


def pose_schema():
    return RequestSchema(
        {
            "pose": FieldSpec(
                "array",
                required=True,
                min_items=3,
                max_items=3,
                items=FieldSpec("number"),
            ),
            "frame": FieldSpec("string", required=True, choices=["workcell"]),
        }
    )


async def echo(context, inputs):
    return inputs


@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"), reason="requires eager task support"
)
@pytest.mark.parametrize("key", [None, "prepared"])
def test_submission_rechecks_closure_after_input_preparation(key):
    async def scenario():
        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        closers, calls = [], []

        class ClosingSchema(RequestSchema):
            def validate(self, content):
                closers.append(asyncio.create_task(engine.aclose()))
                return super().validate(content)

        async def record(context, inputs):
            calls.append(context.node_id)

        engine = WorkflowEngine(
            [Action("record", record, input_schema=ClosingSchema({}))]
        )
        try:
            loop.set_task_factory(asyncio.eager_task_factory)
            with pytest.raises(RuntimeError, match="closed"):
                engine.submit(
                    Workflow((Node("rejected", "record"),)), idempotency_key=key
                )
            await asyncio.wait_for(asyncio.gather(*closers), 1)
            assert calls == []
            assert engine.scheduler_snapshot()["active_runs"] == 0
            assert engine.get_run_by_key("prepared") is None
        finally:
            loop.set_task_factory(previous_factory)
            await engine.aclose(1)

    asyncio.run(scenario())


@pytest.mark.parametrize("key", [None, "prepared"])
def test_submission_rechecks_capacity_after_reentrant_input_preparation(key):
    async def scenario():
        release, entered = asyncio.Event(), asyncio.Event()
        admitted, calls = [], []

        class SubmittingSchema(RequestSchema):
            def validate(self, content):
                if not admitted:
                    admitted.append(engine.submit(Workflow((Node("held", "hold"),))))
                return super().validate(content)

        async def hold(context, inputs):
            entered.set()
            await release.wait()

        async def record(context, inputs):
            calls.append(context.node_id)

        engine = WorkflowEngine(
            [
                Action("hold", hold),
                Action("record", record, input_schema=SubmittingSchema({})),
            ],
            max_active_runs=1,
        )
        try:
            with pytest.raises(RuntimeError, match="max_active_runs"):
                engine.submit(
                    Workflow((Node("rejected", "record"),)), idempotency_key=key
                )
            await asyncio.wait_for(entered.wait(), 1)
            assert engine.scheduler_snapshot()["active_runs"] == 1
            assert engine.get_run_by_key("prepared") is None
            assert calls == []
            release.set()
            assert (await admitted[0].wait(1)).status is RunStatus.SUCCEEDED
        finally:
            release.set()
            await engine.aclose(1)

    asyncio.run(scenario())


def test_integer_subclass_priority_is_captured_before_scheduler_admission():
    class Priority(int):
        def __neg__(self):
            raise RuntimeError("custom queue priority")

        def __int__(self):
            raise RuntimeError("custom integer conversion")

    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)

        urgent = Node("urgent", "record", priority=Priority(7))
        workflow = Workflow((Node("normal", "record"), urgent))
        async with WorkflowEngine(
            [Action("record", record)], max_concurrency=1
        ) as engine:
            assert type(urgent.priority) is int
            run = engine.submit(workflow, idempotency_key="captured-priority")
            assert (await run.wait(1)).status is RunStatus.SUCCEEDED
            assert calls == ["urgent", "normal"]

    asyncio.run(scenario())


def test_resource_quantities_capture_base_integers_before_reservation_arithmetic():
    class Quantity(int):
        def __int__(self):
            raise RuntimeError("custom integer conversion")

        def __add__(self, other):
            raise RuntimeError("custom reservation arithmetic")

        __radd__ = __sub__ = __rsub__ = __add__

        def __lt__(self, other):
            raise RuntimeError("custom positivity comparison")

    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)

        resource = Resource("arm", Quantity(2))
        claim = ResourceClaim("arm", Quantity(1))
        override = ResourceClaim("arm", Quantity(2))
        async with WorkflowEngine(
            [Action("record", record, (claim,))], [resource], max_concurrency=1
        ) as engine:
            assert type(resource.capacity) is int
            assert type(claim.units) is int
            assert type(override.units) is int
            run = engine.submit(
                Workflow(
                    (
                        Node("first", "record"),
                        Node("second", "record", claims=(override,)),
                    )
                ),
                idempotency_key="captured-claims",
            )
            assert (await run.wait(1)).status is RunStatus.SUCCEEDED
            assert calls == ["first", "second"]
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
            assert engine.scheduler_snapshot()["running_actions"] == 0
            assert engine.scheduler_snapshot()["scheduler_error"] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize(
    "definition",
    [
        lambda value: Resource("arm", value),
        lambda value: ResourceClaim("arm", value),
        lambda value: WorkflowEngine([], max_concurrency=value),
        lambda value: WorkflowEngine([], max_input_bytes_per_node=value),
    ],
    ids=["capacity", "claim", "concurrency", "payload-limit"],
)
def test_integer_comparison_hooks_cannot_bypass_positive_quantity_checks(
    value, definition
):
    class FalsePositive(int):
        def __lt__(self, other):
            return False

    with pytest.raises(ValueError, match="positive integer"):
        definition(FalsePositive(value))


@pytest.mark.parametrize("base", [int, float])
@pytest.mark.parametrize("node_stop", [False, True])
def test_timer_budgets_capture_plain_numbers_without_changing_integral_types(
    base, node_stop
):
    class Budget(base):
        def __radd__(self, other):
            raise RuntimeError("custom timer arithmetic")

    class Delay(float):
        def __radd__(self, other):
            raise RuntimeError("custom delay arithmetic")

    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append("entered")
            await context.sleep(Delay(0.001))
            context._request_cancel("cancelled")
            calls.append("stopping")
            return ActionOutcome.cancelled("device stopped")

        node = Node(
            "record",
            "record",
            timeout=Budget(1),
            stop_timeout=Budget(1) if node_stop else None,
        )
        workflow = Workflow((node,), timeout=Budget(1))
        assert type(node.timeout) is base
        if node_stop:
            assert type(node.stop_timeout) is base
        else:
            assert node.stop_timeout is None
        assert type(workflow.timeout) is base
        engine = WorkflowEngine([Action("record", record)], stop_timeout=Budget(1))
        try:
            run = engine.submit(workflow, idempotency_key="captured-budgets")
            assert (await run.wait(Budget(1))).status is RunStatus.CANCELLED
            assert calls == ["entered", "stopping"]
            assert engine.scheduler_snapshot()["scheduler_error"] is None
            assert engine.scheduler_snapshot()["running_actions"] == 0
            assert engine.submit(workflow, idempotency_key="captured-budgets") is run
        finally:
            await engine.aclose(Budget(1))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field",
    [
        "max_concurrency",
        "max_pending_feedback_per_action",
        "max_input_bytes_per_node",
        "max_output_bytes_per_node",
        "max_feedback_bytes_per_report",
    ],
)
def test_async_limits_capture_base_integers_before_comparing_or_charging(field):
    class Limit(int):
        def __int__(self):
            raise RuntimeError("custom limit conversion")

        def __gt__(self, other):
            raise RuntimeError("custom limit operation")

        __eq__ = __sub__ = __gt__

    async def scenario():
        async def record(context, inputs):
            producer = threading.Thread(target=lambda: context.report({"ok": True}))
            producer.start()
            producer.join(1)
            assert not producer.is_alive()
            return inputs

        value = 1 if field == "max_pending_feedback_per_action" else 1024
        async with WorkflowEngine(
            [Action("record", record)], **{field: Limit(value)}
        ) as engine:
            run = engine.submit(Workflow((Node("record", "record", {"ok": True}),)))
            result = await run.wait(1)
            assert result.status is RunStatus.SUCCEEDED
            assert result.nodes["record"].output == {"ok": True}
            assert result.nodes["record"].feedback_dropped == 0
            assert [
                event.detail for event in run.events() if event.kind == "feedback"
            ] == [{"ok": True}]
            assert engine.scheduler_snapshot()["scheduler_error"] is None

    asyncio.run(scenario())


def test_schema_mutated_output_reference_is_rechecked_before_admission():
    class MutatingSchema(RequestSchema):
        def validate_template(self, values, *, is_deferred):
            values["payload"]["ref"] = OutputRef("source", ("missing",))

    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [
                Action(
                    "source",
                    record,
                    output_schema=RequestSchema({"ok": FieldSpec("number")}),
                ),
                Action("sink", record, input_schema=MutatingSchema({})),
            ]
        ) as engine:
            workflow = Workflow(
                (
                    Node("source", "source"),
                    Node(
                        "sink",
                        "sink",
                        {"payload": {"ref": OutputRef("source", ("ok",))}},
                    ),
                )
            )
            with pytest.raises(ValueError, match="invalid output reference"):
                engine.submit(workflow, idempotency_key="unchanged")
            assert engine.get_run_by_key("unchanged") is None
            assert calls == []

    asyncio.run(scenario())


def test_schema_rewritten_binding_waits_for_its_final_source():
    class RebindingSchema(RequestSchema):
        def validate_template(self, values, *, is_deferred):
            values["source"] = OutputRef("new")

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(context, inputs):
            entered.set()
            await release.wait()
            return {"tag": "new"}

        async with WorkflowEngine(
            [
                Action("echo", echo),
                Action("delayed", delayed),
                Action(
                    "sink", echo, input_schema=RebindingSchema({}, allow_extra=True)
                ),
            ]
        ) as engine:
            workflow = Workflow(
                (
                    Node("old", "echo", {"tag": "old"}),
                    Node("sink", "sink", {"source": OutputRef("old")}),
                    Node("new", "delayed"),
                )
            )
            try:
                preview = engine.validate(workflow)
                assert next(
                    node for node in preview["nodes"] if node["node_id"] == "sink"
                )["after"] == ["new"]
                run = engine.submit(workflow)
                await asyncio.wait_for(entered.wait(), 1)
                assert run.snapshot().nodes["sink"].status is NodeStatus.PENDING
                release.set()
                result = await run.wait(1)
                assert result.status is RunStatus.SUCCEEDED
                assert result.nodes["sink"].output == {"source": {"tag": "new"}}
            finally:
                release.set()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "source, diagnostic", [("sink", "cycle"), ("missing", "unknown dependency")]
)
def test_schema_binding_rewrites_are_revalidated_before_key_and_device_admission(
    source, diagnostic
):
    class RebindingSchema(RequestSchema):
        def validate_template(self, values, *, is_deferred):
            values["source"] = OutputRef(source)

    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [
                Action("source", record),
                Action("sink", record, input_schema=RebindingSchema({})),
            ]
        ) as engine:
            with pytest.raises(ValueError, match=diagnostic):
                engine.submit(
                    Workflow(
                        (
                            Node("old", "source"),
                            Node("sink", "sink", {"source": OutputRef("old")}),
                        )
                    ),
                    idempotency_key="schema-binding",
                )
            assert engine.get_run_by_key("schema-binding") is None
            assert calls == []

    asyncio.run(scenario())


def test_valid_output_binds_to_downstream_and_contracts_are_defensive():
    async def scenario():
        schema = pose_schema()
        async with WorkflowEngine(
            [
                Action("detect", echo, output_schema=schema),
                Action(
                    "move",
                    echo,
                    input_schema=RequestSchema(
                        {
                            "pose": FieldSpec(
                                "array",
                                required=True,
                                min_items=3,
                                max_items=3,
                                items=FieldSpec("number"),
                            ),
                        }
                    ),
                ),
            ]
        ) as engine:
            schema._fields["pose"].max_items = 0
            catalog = engine.describe()
            assert (
                catalog["actions"]["detect"]["output_schema"]["fields"]["pose"][
                    "max_items"
                ]
                == 3
            )
            catalog["actions"]["detect"]["output_schema"]["fields"]["pose"][
                "max_items"
            ] = 0
            workflow = Workflow(
                (
                    Node("detect", "detect", {"pose": [1, 2, 3], "frame": "workcell"}),
                    Node("move", "move", {"pose": OutputRef("detect", ("pose",))}),
                )
            )
            run = engine.submit(workflow)
            result = await run.wait(1)
            assert result.status == RunStatus.SUCCEEDED
            assert result.nodes["move"].output == {"pose": [1, 2, 3]}
            result.nodes["detect"].output["pose"].clear()
            assert run.snapshot().nodes["detect"].output["pose"] == [1, 2, 3]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "output",
    [
        None,
        [],
        False,
        {"pose": [1, 2, 3]},
        {"pose": [1, 2], "frame": "workcell"},
        {"pose": [1, "bad", 3], "frame": "workcell"},
        {"pose": [1, 2, 3], "frame": "camera"},
    ],
)
def test_invalid_output_fails_producer_and_never_dispatches_dependent_action(output):
    async def scenario():
        calls = []

        async def detect(context, inputs):
            calls.append("detect")
            return output

        async def move(context, inputs):
            calls.append("move")

        async with WorkflowEngine(
            [
                Action(
                    "detect",
                    detect,
                    (ResourceClaim("camera"),),
                    output_schema=pose_schema(),
                ),
                Action("move", move),
            ],
            [Resource("camera")],
        ) as engine:
            workflow = Workflow(
                (
                    Node("detect", "detect"),
                    Node("move", "move", {"pose": OutputRef("detect", ("pose",))}),
                )
            )
            run = engine.submit(workflow, idempotency_key="cycle")
            result = await run.wait(1)
            assert result.status == RunStatus.FAILED
            assert result.nodes["detect"].status == NodeStatus.FAILED
            assert result.nodes["detect"].error.startswith("ActionOutputError:")
            assert result.nodes["detect"].output == output
            assert result.nodes["move"].status == NodeStatus.SKIPPED
            assert calls == ["detect"]
            assert engine.resource_snapshot()["resources"]["camera"]["used"] == 0
            assert not engine.resource_snapshot()["quarantined"]
            assert engine.submit(workflow, idempotency_key="cycle") is run

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "path", [("missing",), (0,), ("pose", 3), ("pose", "x"), ("frame", 0)]
)
def test_known_invalid_output_reference_is_rejected_before_any_device_io(path):
    async def scenario():
        calls = []

        async def detect(context, inputs):
            calls.append("detect")
            return {"pose": [1, 2, 3], "frame": "workcell"}

        async with WorkflowEngine(
            [
                Action("detect", detect, output_schema=pose_schema()),
                Action("echo", echo),
            ]
        ) as engine:
            workflow = Workflow(
                (
                    Node("detect", "detect"),
                    Node("use", "echo", {"value": OutputRef("detect", path)}),
                )
            )
            with pytest.raises(ValueError, match="invalid output reference"):
                engine.validate(workflow)
            with pytest.raises(ValueError, match="invalid output reference"):
                engine.submit(workflow, idempotency_key="invalid")
            assert engine.get_run_by_key("invalid") is None
            assert not calls

    asyncio.run(scenario())


def test_optional_output_fields_remain_a_runtime_check():
    async def scenario():
        async with WorkflowEngine(
            [
                Action(
                    "detect",
                    echo,
                    output_schema=RequestSchema({"pose": FieldSpec("array")}),
                ),
                Action("echo", echo),
            ]
        ) as engine:
            workflow = Workflow(
                (
                    Node("detect", "detect"),
                    Node("use", "echo", {"value": OutputRef("detect", ("pose",))}),
                )
            )
            engine.validate(workflow)
            result = await engine.submit(workflow).wait(1)
            assert result.nodes["detect"].status == NodeStatus.SUCCEEDED
            assert result.nodes["use"].status == NodeStatus.FAILED
            assert "KeyError" in result.nodes["use"].error

    asyncio.run(scenario())


def test_unknown_operation_keeps_claims_without_output_validation():
    async def scenario():
        async def lost(context, inputs):
            raise ExecutionUncertain("controller state unavailable")

        async with WorkflowEngine(
            [
                Action(
                    "move", lost, (ResourceClaim("arm"),), output_schema=pose_schema()
                ),
            ],
            [Resource("arm")],
        ) as engine:
            run = engine.submit(Workflow((Node("n", "move"),)))
            result = await run.wait(1)
            assert result.status == RunStatus.NEEDS_ATTENTION
            assert result.nodes["n"].status == NodeStatus.UNKNOWN
            assert "ExecutionUncertain" in result.nodes["n"].error
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            engine.confirm_quiescent(run.run_id + "/n", evidence="arm stopped")

    asyncio.run(scenario())


def test_output_failure_preserves_independent_branches():
    async def scenario():
        async with WorkflowEngine(
            [
                Action("invalid", echo, output_schema=pose_schema()),
                Action("echo", echo),
            ]
        ) as engine:
            result = await engine.submit(
                Workflow(
                    (
                        Node("bad", "invalid"),
                        Node("dependent", "echo", after=("bad",)),
                        Node("independent", "echo", {"spoken": True}),
                    ),
                    FailurePolicy.CONTINUE_INDEPENDENT,
                )
            ).wait(1)
            assert result.nodes["bad"].status == NodeStatus.FAILED
            assert result.nodes["dependent"].status == NodeStatus.SKIPPED
            assert result.nodes["independent"].status == NodeStatus.SUCCEEDED

    asyncio.run(scenario())


def test_empty_object_schema_does_not_coerce_none_to_a_successful_result():
    async def scenario():
        async def empty(context, inputs):
            return None

        async with WorkflowEngine(
            [Action("empty", empty, output_schema=RequestSchema({}))]
        ) as engine:
            result = await engine.submit(Workflow((Node("n", "empty"),))).wait(1)
            assert result.status == RunStatus.FAILED
            assert "must be a dictionary" in result.nodes["n"].error

    asyncio.run(scenario())


def test_output_contract_configuration_requires_a_schema():
    with pytest.raises(TypeError, match="output_schema"):
        Action("echo", echo, output_schema={})


def test_schema_checked_at_admission_and_again_after_output_binding():
    async def scenario():
        calls = []

        async def device(context, inputs):
            calls.append(inputs)

        schema = RequestSchema({"x": FieldSpec("number", required=True)})
        async with WorkflowEngine(
            [Action("device", device, input_schema=schema), Action("echo", echo)]
        ) as engine:
            with pytest.raises(CommandValidationError):
                engine.submit(Workflow((Node("bad", "device", {"x": "wrong"}),)))
            run = engine.submit(
                Workflow(
                    (
                        Node("source", "echo", {"x": "wrong"}),
                        Node("sink", "device", {"x": OutputRef("source", ("x",))}),
                    )
                )
            )
            result = await run.wait(1)
            assert result.nodes["source"].status == NodeStatus.SUCCEEDED
            assert result.nodes["sink"].status == NodeStatus.FAILED
            assert calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "inputs, message",
    [
        pytest.param(
            {"pose": OutputRef("detect", ("pose",))},
            "Missing required field: 'config'", id="missing-required",
        ),
        pytest.param(
            {"pose": OutputRef("detect"), "config": {}, "extra": True},
            "Unknown command fields: extra", id="unknown-field",
        ),
        pytest.param(
            {"pose": OutputRef("detect"), "config": "fast"},
            "Field 'config' must be object", id="container-type",
        ),
        pytest.param(
            {"pose": OutputRef("detect"), "config": {"speed": 2}},
            "Field 'config.speed' must be at most 1", id="literal-range",
        ),
        pytest.param(
            {"pose": OutputRef("detect"), "config": {"speed": True}},
            "Field 'config.speed' must be number", id="literal-type",
        ),
        pytest.param(
            {"pose": [1, 2, 3], "config": {"speed": OutputRef("detect")}},
            "Missing required field: 'config.mode'", id="nested-required",
        ),
        pytest.param(
            {"pose": [1, 2, 3], "config": {"speed": OutputRef("detect"), "typo": 1}},
            "Unknown command fields: config.typo", id="nested-unknown",
        ),
        pytest.param(
            {"pose": [1, 2, 3], "config": {"speed": OutputRef("detect"), "mode": "fast"}},
            "Field 'config.mode' must be one of", id="literal-choice",
        ),
        pytest.param(
            {"pose": [OutputRef("detect"), True, 3], "config": OutputRef("detect")},
            "Field 'pose[1]' must be number", id="array-item-type",
        ),
        pytest.param(
            {"pose": [OutputRef("detect"), -1, 3], "config": OutputRef("detect")},
            "Field 'pose[1]' must be at least 0", id="array-item-range",
        ),
        pytest.param(
            {"pose": [OutputRef("detect"), 2], "config": OutputRef("detect")},
            "Field 'pose' requires at least 3 items", id="array-length",
        ),
    ],
)
def test_mixed_inputs_reject_known_errors_before_any_device_dispatch(inputs, message):
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)
            return inputs

        schema = RequestSchema(
            {
                "pose": FieldSpec(
                    "array", required=True, min_items=3, max_items=3,
                    items=FieldSpec("number", minimum=0),
                ),
                "config": FieldSpec(
                    "object", required=True,
                    object_schema=RequestSchema(
                        {
                            "speed": FieldSpec("number", required=True, maximum=1),
                            "mode": FieldSpec("string", required=True, choices=["safe"]),
                        }
                    ),
                ),
            }
        )
        async with WorkflowEngine(
            [Action("detect", record), Action("move", record, input_schema=schema)],
            max_active_runs=1, max_idempotency_keys=1,
        ) as engine:
            workflow = Workflow((Node("detect", "detect"), Node("move", "move", inputs)))
            for operation in (
                lambda: engine.validate(workflow),
                lambda: engine.submit(workflow, idempotency_key="cycle"),
            ):
                with pytest.raises(CommandValidationError) as error:
                    operation()
                assert message in str(error.value)
            assert engine.get_run_by_key("cycle") is None
            await asyncio.sleep(0)
            assert not calls
            healthy = engine.submit(
                Workflow((Node("healthy", "detect"),)), idempotency_key="cycle"
            )
            assert (await healthy.wait(1)).status == RunStatus.SUCCEEDED
            assert calls == ["healthy"]

    asyncio.run(scenario())


@pytest.mark.parametrize("shape", ["array", "object"])
@pytest.mark.parametrize("resolved", [1, 2, True])
def test_composite_choices_validate_literals_and_recheck_bound_values(shape, resolved):
    async def scenario():
        calls = []

        def wrap(value, mode="safe"):
            return [value, mode] if shape == "array" else {"value": value, "mode": mode}

        async def source(context, inputs):
            calls.append("detect")
            return {"value": resolved}

        async def device(context, inputs):
            calls.append("move")
            return inputs

        async with WorkflowEngine(
            [
                Action("detect", source),
                Action(
                    "move", device, (ResourceClaim("arm"),),
                    input_schema=RequestSchema(
                        {"box": FieldSpec(shape, required=True, choices=[wrap(1)])}
                    ),
                ),
            ],
            [Resource("arm")],
        ) as engine:
            reference = OutputRef("detect", ("value",))
            for entrypoint in (engine.validate, engine.submit):
                with pytest.raises(CommandValidationError, match="must be one of"):
                    entrypoint(Workflow((
                        Node("detect", "detect"),
                        Node("move", "move", {"box": wrap(reference, "bad")}),
                    )))
            workflow = Workflow((
                Node("detect", "detect"),
                Node("move", "move", {"box": wrap(reference)}),
            ))
            engine.validate(workflow)
            run = engine.submit(workflow)
            result = await run.wait(1)
            if type(resolved) is int and resolved == 1:
                assert result.status == RunStatus.SUCCEEDED
                assert result.nodes["move"].output == {"box": wrap(1)}
                assert calls == ["detect", "move"]
            else:
                assert result.nodes["move"].status == NodeStatus.FAILED
                assert "must be one of" in result.nodes["move"].error
                assert calls == ["detect"]
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "path, category",
    [
        (("missing",), "KeyError"),
        (("list", 8), "IndexError"),
        (("list", "wrong"), "ValueError"),
        ((0,), "ValueError"),
        (("list", 0, "nested"), "ValueError"),
    ],
)
def test_bad_output_path_fails_node_without_device_io(path, category):
    async def scenario():
        calls = []

        async def consume(context, inputs):
            calls.append(context.node_id)
            return inputs

        async with WorkflowEngine(
            [Action("echo", echo), Action("consume", consume, (ResourceClaim("arm"),))],
            [Resource("arm")],
        ) as engine:
            run = engine.submit(
                Workflow(
                    (
                        Node("source", "echo", {"list": [1]}),
                        Node("sink", "consume", {"value": OutputRef("source", path)}),
                    )
                )
            )
            result = await run.wait(1)
            assert result.nodes["sink"].status == NodeStatus.FAILED
            error = result.nodes["sink"].error
            assert error.startswith(category + ":")
            assert "source" in error
            # KeyError formats its message with repr, which may escape quotes.
            assert "path" in error
            for part in path:
                assert str(part) in error
            assert not calls
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0
            assert not engine.resource_snapshot()["quarantined"]
            follow_up = engine.submit(Workflow((Node("next", "consume"),)))
            assert (await follow_up.wait(1)).status is RunStatus.SUCCEEDED
            assert calls == ["next"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), {1: "bad"}, b"frame", object()]
)
def test_outputs_must_be_small_json_values_or_artifact_references(value):
    async def scenario():
        async def invalid(context, inputs):
            return value

        async with WorkflowEngine([Action("invalid", invalid)]) as engine:
            run = engine.submit(Workflow((Node("bad", "invalid"),)))
            assert (await run.wait(1)).nodes["bad"].status == NodeStatus.FAILED

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Resource(""),
        lambda: Resource("gpu", True),
        lambda: Resource("gpu", 0),
        lambda: ResourceClaim("arm", -1),
        lambda: ResourceClaim("arm", 1.5),
        lambda: Node("x", "echo", timeout=float("inf")),
        lambda: Node("x", "echo", timeout=True),
        lambda: Node("x", "echo", timeout=0),
        lambda: Node("x", "echo", priority=True),
        lambda: Node("x", "echo", after="y"),
        lambda: Node("x", "echo", inputs={1: "bad"}),
        lambda: OutputRef("x", "field"),
        lambda: OutputRef("x", (-1,)),
        lambda: OutputRef("x", (True,)),
        lambda: Workflow(()),
        lambda: WorkflowEngine([], max_concurrency=0),
        lambda: WorkflowEngine([], max_events_per_run=True),
        lambda: WorkflowEngine([Action("x", echo), Action("x", echo)]),
        lambda: WorkflowEngine([], [Resource("x"), Resource("x")]),
        lambda: WorkflowEngine([Action("x", echo, (ResourceClaim("missing"),))]),
        lambda: Action("sync", lambda context, inputs: None),
        lambda: BlockingAction(echo),
    ],
)
def test_invalid_definitions_are_rejected(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()
