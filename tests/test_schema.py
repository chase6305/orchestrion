import concurrent.futures
import json

import pytest

from orchestrion import (
    CallableTask,
    CommandValidationError,
    FieldSpec,
    MoveSyncOption,
    RequestSchema,
    RequestStatus,
    SimulatedRobotTask,
    TaskPilot,
)


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


def test_callable_schema_failure_never_calls_or_retries_device():
    device_calls = []

    def capture(request_id, content):
        device_calls.append(content)
        return {"captured": True}

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    task = CallableTask(capture, request_schema=make_camera_schema())
    pilot = TaskPilot(
        SimulatedRobotTask([0.0]), {"camera": task}, executor_owned=executor
    )
    pilot.initialize()
    try:
        request_id = pilot.call_srv_async(
            "camera", {"exposure_ms": -1}, MoveSyncOption.no_sync()
        )
        result = pilot.wait_request(request_id, timeout=0.5)
        assert result.status is RequestStatus.FAILED
        assert "at least" in result.error
        assert device_calls == []
        assert task.peek_status()["total_retries"] == 0
    finally:
        pilot.stop()


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
