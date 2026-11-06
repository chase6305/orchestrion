"""Workflow definitions, input/output contracts, and downstream data binding."""

import asyncio

import pytest

from orchestrion import CommandValidationError, FieldSpec, RequestSchema
from orchestrion.workflow import (
    Action,
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


@pytest.mark.parametrize("last_field", ["frame", "missing"])
def test_nested_forward_references_validate_every_path_before_dispatch(last_field):
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)
            return inputs

        async with WorkflowEngine(
            [
                Action("detect", record, output_schema=pose_schema()),
                Action("use", record),
            ]
        ) as engine:
            workflow = Workflow(
                (
                    Node(
                        "use",
                        "use",
                        {
                            "nested": [
                                {
                                    "first": OutputRef("detect", ("pose", 0)),
                                    "last": OutputRef("detect", (last_field,)),
                                }
                            ]
                        },
                    ),
                    Node("detect", "detect", {"pose": [1, 2, 3], "frame": "workcell"}),
                )
            )
            if last_field == "missing":
                with pytest.raises(ValueError, match="invalid output reference"):
                    engine.submit(workflow)
                assert not calls
            else:
                result = await engine.submit(workflow).wait(1)
                assert result.status == RunStatus.SUCCEEDED
                assert result.nodes["use"].output == {
                    "nested": [{"first": 1, "last": "workcell"}]
                }
                assert calls == ["detect", "use"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "target, field, error",
    [
        ("source", "other", None),
        ("source", "missing", "invalid output reference"),
        ("replacement", "other", None),
        ("absent", "other", "unknown dependency"),
    ],
)
def test_references_changed_by_schema_hooks_are_rechecked_before_dispatch(
    target, field, error,
):
    class ChangedTemplate(RequestSchema):
        def validate_template(self, content, *, is_deferred):
            content["values"][0] = OutputRef(target, (field,))
            super().validate_template(content, is_deferred=is_deferred)

    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(context.node_id)
            return inputs

        async with WorkflowEngine(
            [
                Action(
                    "source", record,
                    output_schema=RequestSchema({
                        "value": FieldSpec("number"), "other": FieldSpec("number"),
                    }),
                ),
                Action(
                    "use", record,
                    input_schema=ChangedTemplate({
                        "values": FieldSpec("array", items=FieldSpec("number")),
                    }),
                ),
            ],
            max_concurrency=1,
        ) as engine:
            reference = OutputRef("source", ("value",))
            workflow = Workflow((
                Node("use", "use", {"values": [reference]}),
                Node("source", "source", {"value": 1, "other": 2}),
                Node(
                    "replacement", "source", {"value": 3, "other": 4}, after=("gate",)
                ),
                Node("gate", "source", {"value": 0, "other": 0}),
            ))
            if error is not None:
                with pytest.raises(ValueError, match=error):
                    engine.submit(workflow)
                assert not calls
            else:
                result = await engine.submit(workflow).wait(1)
                assert result.status == RunStatus.SUCCEEDED
                expected = 4 if target == "replacement" else 2
                assert result.nodes["use"].output == {"values": [expected]}
                assert calls.index(target) < calls.index("use")
            assert workflow.nodes[0].inputs == {"values": [reference]}

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
