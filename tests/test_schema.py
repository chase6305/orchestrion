"""Flat and nested command schemas, defensive copies, and reference paths."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from orchestrion import (
    CallableTask,
    CommandValidationError,
    FieldSpec,
    MoveSyncOption,
    RequestSchema,
    RequestStatus,
    RetryPolicy,
    SimulatedRobotTask,
    TaskPilot,
)
from orchestrion.utils.types import PeekResponseResultType


def make_camera_schema():
    return RequestSchema(
        {
            "exposure_ms": FieldSpec(
                "number",
                required=True,
                minimum=0.1,
                maximum=100.0,
                description="Camera exposure in milliseconds",
            ),
            "trigger": FieldSpec(
                "string", choices=["software", "hardware"]
            ),
            "hdr": FieldSpec("boolean"),
        }
    )


def detection_schema():
    return RequestSchema(
        {
            "objects": FieldSpec(
                "array",
                required=True,
                min_items=1,
                max_items=2,
                items=FieldSpec(
                    "object",
                    object_schema=RequestSchema(
                        {
                            "pose": FieldSpec(
                                "array",
                                required=True,
                                min_items=3,
                                max_items=3,
                                items=FieldSpec("number"),
                            ),
                            "confidence": FieldSpec(
                                "number", required=True, minimum=0, maximum=1
                            ),
                        }
                    ),
                ),
            ),
        }
    )


def test_request_schema_validates_and_copies_payload():
    schema = make_camera_schema()
    content = {"exposure_ms": 8, "trigger": "software", "hdr": True}
    validated = schema.validate(content)
    assert validated == content
    assert validated is not content
    json.dumps(schema.describe())


@pytest.mark.parametrize(
    "content, message",
    [
        ({}, "Missing required"),
        ({"exposure_ms": True}, "must be number"),
        ({"exposure_ms": 0.01}, "at least"),
        ({"exposure_ms": 101}, "at most"),
        ({"exposure_ms": 8, "trigger": "timer"}, "one of"),
        ({"exposure_ms": 8, "unknown": 1}, "Unknown command fields"),
        ({1: "bad", "exposure_ms": 8}, "field names"),
    ],
)
def test_request_schema_rejects_invalid_payload(content, message):
    with pytest.raises(CommandValidationError, match=message):
        make_camera_schema().validate(content)


@pytest.mark.parametrize(
    "copied, message",
    [
        ({"exposure_ms": -1}, "at least"),
        ({"exposure_ms": 8, "unknown": True}, "Unknown command fields"),
        (None, "must be a dictionary"),
    ],
)
def test_schema_checks_copied_payload_before_device_execution(copied, message):
    class Payload(dict):
        def __deepcopy__(self, memo):
            return copied

    payload = Payload(exposure_ms=8)
    schema = make_camera_schema()
    with pytest.raises(CommandValidationError, match=message):
        schema.validate(payload)

    device_calls = []
    completed = threading.Event()
    task = CallableTask(
        lambda request_id, content: device_calls.append(content),
        request_schema=schema,
        retry_policy=RetryPolicy(max_attempts=3, delay=0),
    )
    task.set_completion_callback(completed.set)
    with ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0, payload)
            assert completed.wait(1)
            response = task.peek_response(0)
            assert response.result_type is PeekResponseResultType.ErrorUnknown
            assert message in response.error
            assert not device_calls
            assert task.peek_status()["total_retries"] == 0
        finally:
            task.stop()


def test_schema_checks_nested_values_created_during_copy():
    class Values(list):
        def __deepcopy__(self, memo):
            return [1, "invalid", 3]

    schema = RequestSchema({"values": FieldSpec("array", items=FieldSpec("number"))})
    with pytest.raises(CommandValidationError, match=r"values\[1\]"):
        schema.validate({"values": Values([1, 2, 3])})


@pytest.mark.parametrize(
    "schema, payload, message",
    [
        pytest.param(make_camera_schema(), {"exposure_ms": -1}, "at least", id="flat"),
        pytest.param(
            detection_schema(),
            {"objects": [{"pose": [1, "bad", 3], "confidence": 0.9}]},
            "objects[0].pose[1]",
            id="nested",
        ),
    ],
)
def test_callable_schema_failure_never_calls_or_retries_device(schema, payload, message):
    device_calls = []
    task = CallableTask(
        lambda request_id, content: device_calls.append(content),
        request_schema=schema,
    )
    with TaskPilot(
        SimulatedRobotTask([0.0]), {"camera": task},
        executor_owned=ThreadPoolExecutor(max_workers=1),
    ) as pilot:
        request_id = pilot.call_srv_async("camera", payload, MoveSyncOption.no_sync())
        result = pilot.wait_request(request_id, timeout=1)
        assert result.status is RequestStatus.FAILED
        assert message in result.error
        assert device_calls == []
        assert task.peek_status()["total_retries"] == 0


def test_schema_is_exposed_through_service_discovery():
    task = CallableTask(
        lambda request_id, content: content, request_schema=make_camera_schema()
    )
    schema = task.describe()["input_schema"]
    assert schema["fields"]["exposure_ms"] == {
        "type": "number",
        "required": True,
        "minimum": 0.1,
        "maximum": 100.0,
        "description": "Camera exposure in milliseconds",
    }


def test_schema_configuration_is_validated():
    with pytest.raises(ValueError, match="Unsupported"):
        FieldSpec("timestamp")
    with pytest.raises(ValueError, match="numeric"):
        FieldSpec("string", minimum=0)
    with pytest.raises(ValueError, match="minimum"):
        FieldSpec("number", minimum=2, maximum=1)
    with pytest.raises(ValueError, match="JSON-compatible"):
        FieldSpec("number", choices=[float("nan")])
    with pytest.raises(ValueError, match="field constraints"):
        FieldSpec("integer", choices=[1.5])
    with pytest.raises(ValueError, match="field constraints"):
        FieldSpec("number", minimum=0, choices=[-1])
    with pytest.raises(TypeError, match="FieldSpec"):
        RequestSchema({"value": object()})
    with pytest.raises(TypeError, match="request_schema"):
        CallableTask(lambda request_id, content: content, request_schema={})


def test_integer_limits_keep_precision_above_double_range():
    lower = 2 ** 53 + 1
    upper = 2 ** 63 - 1
    field = FieldSpec("integer", minimum=lower, maximum=upper)
    field.validate("id", lower)
    field.validate("id", upper)
    with pytest.raises(CommandValidationError, match="at least"):
        field.validate("id", lower - 1)
    with pytest.raises(CommandValidationError, match="at most"):
        field.validate("id", upper + 1)
    assert field.describe()["minimum"] == lower
    assert json.loads(json.dumps(field.describe()))["maximum"] == upper


def test_number_fields_accept_finite_integers_larger_than_float_range():
    value = 10 ** 400
    field = FieldSpec("number", maximum=value)
    field.validate("counter", value)
    with pytest.raises(CommandValidationError):
        field.validate("counter", value + 1)


def test_schema_snapshots_field_constraints_at_construction():
    field = FieldSpec("string", choices=["grip"])
    schema = RequestSchema({"action": field})
    field.choices.append("release")
    with pytest.raises(CommandValidationError, match="one of"):
        schema.validate({"action": "release"})
    assert schema.describe()["fields"]["action"]["choices"] == ["grip"]


def test_nested_schema_validates_and_defensively_copies_data_and_description():
    schema = detection_schema()
    payload = {"objects": [{"pose": [1, 2.0, 3], "confidence": 0.9}]}
    validated = schema.validate(payload)
    payload["objects"][0]["pose"].clear()
    assert validated == {"objects": [{"pose": [1, 2.0, 3], "confidence": 0.9}]}
    description = schema.describe()
    fields = description["fields"]["objects"]["items"]["object_schema"]["fields"]
    assert fields["pose"]["min_items"] == fields["pose"]["max_items"] == 3
    assert fields["pose"]["items"]["type"] == "number"
    fields["pose"]["items"]["type"] = "string"
    assert schema.validate(validated) == validated
    json.dumps(schema.describe(), allow_nan=False)


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"objects": []}, "at least 1 items"),
        ({"objects": [None, None, None]}, "at most 2 items"),
        ({"objects": [None]}, "objects[0]"),
        ({"objects": [{"pose": [1, 2, 3]}]}, "objects[0].confidence"),
        ({"objects": [{"pose": [1, 2], "confidence": 0.9}]}, "objects[0].pose"),
        ({"objects": [{"pose": [1, 2, 3, 4], "confidence": 0.9}]}, "objects[0].pose"),
        (
            {"objects": [{"pose": [1, True, 3], "confidence": 0.9}]},
            "objects[0].pose[1]",
        ),
        (
            {"objects": [{"pose": [1, float("nan"), 3], "confidence": 0.9}]},
            "objects[0].pose[1]",
        ),
        (
            {"objects": [{"pose": [1, float("inf"), 3], "confidence": 0.9}]},
            "objects[0].pose[1]",
        ),
        (
            {"objects": [{"pose": [1, 2, 3], "confidence": 1.1}]},
            "objects[0].confidence",
        ),
        (
            {"objects": [{"pose": [1, 2, 3], "confidence": 0.9, "extra": 1}]},
            "objects[0].extra",
        ),
        ({"objects": [{1: "bad"}]}, "field names must be strings"),
    ],
)
def test_nested_validation_reports_the_failing_path(payload, message):
    with pytest.raises(CommandValidationError) as error:
        detection_schema().validate(payload)
    assert message in str(error.value)


def test_nested_constraints_are_snapshotted_when_attached_to_parent():
    item = FieldSpec("number", minimum=0)
    array = FieldSpec("array", items=item, max_items=2)
    item.minimum = 100
    array.validate("value", [1])
    child = RequestSchema({"values": array})
    parent = FieldSpec("object", object_schema=child)
    child._fields["values"].max_items = 0
    parent.validate("value", {"values": [1]})
    with pytest.raises(CommandValidationError):
        parent.validate("value", {"values": [-1]})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"items": "number"},
        {"object_schema": {}},
        {"min_items": -1},
        {"max_items": -1},
        {"min_items": True},
        {"max_items": False},
        {"min_items": 1.0},
        {"max_items": "2"},
        {"min_items": 2, "max_items": 1},
    ],
)
def test_invalid_array_schema_configuration_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FieldSpec("array", **kwargs)


def test_nested_constraints_require_the_corresponding_field_types():
    with pytest.raises(ValueError, match="array field"):
        FieldSpec("number", items=FieldSpec("number"))
    with pytest.raises(ValueError, match="array field"):
        FieldSpec("object", max_items=1)
    with pytest.raises(ValueError, match="object field"):
        FieldSpec("array", object_schema=RequestSchema({}))
    with pytest.raises(ValueError, match="choices must satisfy"):
        FieldSpec("array", choices=[[1, "bad"]], items=FieldSpec("number"))
    FieldSpec("array", min_items=0, max_items=0).validate("empty", [])


@pytest.mark.parametrize(
    "path",
    [
        ("missing",),
        (0,),
        ("objects", "pose"),
        ("objects", 2),
        ("objects", 0, "missing"),
        ("objects", 0, "pose", 3),
        ("objects", 0, "pose", 0, "x"),
    ],
)
def test_reference_paths_reject_statically_impossible_access(path):
    with pytest.raises(CommandValidationError):
        detection_schema().validate_path(path)


def test_reference_paths_defer_optional_fields_short_arrays_and_open_objects():
    schema = RequestSchema(
        {
            "optional": FieldSpec("array", items=FieldSpec("number"), max_items=3),
            "open": FieldSpec("object"),
            "loose": FieldSpec("array"),
            "extended": FieldSpec(
                "object", object_schema=RequestSchema({}, allow_extra=True)
            ),
        }
    )
    for path in [
        (),
        ("optional", 2),
        ("open", "dynamic", 10),
        ("loose", 100, "dynamic"),
        ("extended", "dynamic", 0),
    ]:
        schema.validate_path(path)
    schema.validate({})


@pytest.mark.parametrize("path", ["value", (True,), (-1,), (1.5,), (None,)])
def test_reference_paths_reject_invalid_components_even_in_open_schemas(path):
    with pytest.raises((TypeError, ValueError)):
        RequestSchema({}, allow_extra=True).validate_path(path)


def test_template_validation_preserves_placeholders_and_requires_full_validation():
    marker = object()
    schema = RequestSchema(
        {"pose": FieldSpec("array", required=True, items=FieldSpec("number"), max_items=3)}
    )
    payload = {"pose": [marker, 2, 3]}
    def is_deferred(value):
        return value is marker

    assert schema.validate_template(payload, is_deferred=is_deferred) is None
    assert payload["pose"][0] is marker
    with pytest.raises(CommandValidationError, match="must be number"):
        schema.validate(payload)
    resolved = {"pose": [1, 2, 3]}
    validated = schema.validate(resolved)
    resolved["pose"].clear()
    assert validated == {"pose": [1, 2, 3]}


@pytest.mark.parametrize("content", [None, [], "value", {1: "value"}])
def test_template_validation_requires_an_object_with_string_keys(content):
    with pytest.raises(CommandValidationError):
        RequestSchema({}).validate_template(content, is_deferred=lambda value: False)


def test_template_validation_requires_an_explicit_callable_predicate():
    with pytest.raises(TypeError, match="is_deferred must be callable"):
        RequestSchema({}).validate_template({}, is_deferred=None)


@pytest.mark.parametrize(
    "value_type, choice, invalid",
    [
        ("array", [True], [1]),
        ("array", [0], [False]),
        ("object", {"stopped": True}, {"stopped": 1}),
        ("object", {"pose": [1]}, {"pose": [True]}),
    ],
)
def test_composite_choices_distinguish_json_booleans_from_numbers(value_type, choice, invalid):
    spec = FieldSpec(value_type, choices=[choice])
    spec.validate("result", choice)
    with pytest.raises(CommandValidationError, match="must be one of"):
        spec.validate("result", invalid)


@pytest.mark.parametrize(
    "value_type, choice, value",
    [("array", [1], [1.0]), ("object", {"x": [1]}, {"x": (1.0,)})],
)
def test_composite_choices_compare_numeric_and_array_values_as_json(value_type, choice, value):
    FieldSpec(value_type, choices=[choice]).validate("result", value)


@pytest.mark.parametrize("value_type", ["array", "object"])
def test_choice_construction_preserves_nested_enumeration_constraints(value_type):
    item = FieldSpec("integer", choices=[1, 2])
    if value_type == "array":
        constraints = {"items": item}
        allowed, disallowed = [1], [3]
    else:
        constraints = {"object_schema": RequestSchema({"value": item})}
        allowed, disallowed = {"value": 1}, {"value": 3}

    spec = FieldSpec(value_type, choices=[allowed], **constraints)
    spec.validate("result", allowed)
    # A later candidate must still satisfy the nested field's own enumeration.
    with pytest.raises(ValueError, match="choices must satisfy"):
        FieldSpec(value_type, choices=[allowed, disallowed], **constraints)
    with pytest.raises(CommandValidationError, match="must be one of"):
        spec.validate("result", disallowed)


def test_choice_construction_keeps_isolated_nested_values():
    choices = [{"pose": [1, 2]}]
    spec = FieldSpec("object", choices=iter(choices))
    choices[0]["pose"].append(3)
    spec.validate("result", {"pose": [1, 2]})
    with pytest.raises(CommandValidationError, match="must be one of"):
        spec.validate("result", choices[0])


@pytest.mark.parametrize("value_type, value", [("integer", 1), ("array", [])])
def test_empty_choices_remain_an_empty_enumeration(value_type, value):
    spec = FieldSpec(value_type, choices=[])
    assert spec.describe()["choices"] == []
    with pytest.raises(CommandValidationError, match="must be one of"):
        spec.validate("result", value)


@pytest.mark.parametrize("value_type", ["integer", "object"])
def test_choice_construction_avoids_quadratic_candidate_comparisons(value_type):
    comparisons = 0

    class CountedInteger(int):
        def __eq__(self, other):
            nonlocal comparisons
            comparisons += 1
            return super().__eq__(other)

        __hash__ = int.__hash__

    choices = [
        CountedInteger(index) if value_type == "integer"
        else {"value": CountedInteger(index)}
        for index in range(100)
    ]
    spec = FieldSpec(value_type, choices=choices)
    # Count comparisons rather than imposing a machine-dependent time limit.
    assert comparisons <= 2 * len(choices)
    spec.validate("result", choices[-1])
