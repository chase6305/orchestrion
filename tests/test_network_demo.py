import json
import subprocess
import sys
import time

import pytest

from examples.network_device_service import (
    IdempotencyConflictError,
    NetworkDeviceServer,
    SimulatedNetworkDevice,
)
from examples.network_service_demo import NetworkDeviceClient, run_workflow


def test_network_device_deduplicates_ticketed_commands():
    device = SimulatedNetworkDevice()
    command = {"action": "grip", "duration": 0.001}
    first = device.submit(command, "cycle-1/grip")
    second = device.submit(command, "cycle-1/grip")
    assert first["operation_id"] == second["operation_id"]
    assert not first["deduplicated"]
    assert second["deduplicated"]
    result = device.wait(first["operation_id"], timeout=0.5)
    assert result["status"] == "succeeded"
    assert result["result"] == {"action": "grip", "accepted": True}


def test_network_device_rejects_idempotency_key_reuse_with_new_command():
    device = SimulatedNetworkDevice()
    device.submit({"action": "grip", "duration": 0.001}, "cycle-1/tool")
    with pytest.raises(IdempotencyConflictError, match="different command"):
        device.submit({"action": "release", "duration": 0.001}, "cycle-1/tool")


def test_network_workflow_waits_for_remote_operation():
    with NetworkDeviceServer() as server:
        summary = run_workflow(server.base_url)
    assert summary["result"]["action"] == "grip"
    assert summary["result"]["accepted"]
    assert summary["timeline"] == [
        "queued",
        "waiting_for_move",
        "running",
        "succeeded",
    ]
    assert summary["service"]["metadata"]["protocol"] == "ticket-and-long-poll"
    assert summary["health"]["running_operations"] == 0


def test_network_client_health_endpoint():
    with NetworkDeviceServer() as server:
        status = NetworkDeviceClient(server.base_url).status()
    assert status["health"] == "online"
    assert status["available"]


@pytest.mark.parametrize("base_url", ["", "   ", None])
def test_network_client_rejects_invalid_base_url(base_url):
    with pytest.raises(ValueError, match="base_url"):
        NetworkDeviceClient(base_url)


@pytest.mark.parametrize(
    "request_timeout", [0, -1, float("nan"), float("inf"), "0.5", True]
)
def test_network_client_rejects_invalid_request_timeout(request_timeout):
    with pytest.raises(ValueError, match="request_timeout"):
        NetworkDeviceClient("http://127.0.0.1", request_timeout=request_timeout)


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "unknown", "duration": 0.01},
        {"action": "grip", "duration": -1},
        {"action": "grip", "duration": float("nan")},
        {"action": "grip", "duration": True},
        {"action": "grip", "duration": 0.01, "extra": True},
    ],
)
def test_network_server_rejects_invalid_commands(payload):
    with NetworkDeviceServer() as server:
        client = NetworkDeviceClient(server.base_url)
        with pytest.raises(RuntimeError, match="HTTP 400"):
            client._request_json(
                "/commands",
                method="POST",
                payload=payload,
                headers={"Idempotency-Key": "invalid-command"},
            )


def test_network_server_reports_http_idempotency_conflict():
    with NetworkDeviceServer() as server:
        client = NetworkDeviceClient(server.base_url)
        headers = {"Idempotency-Key": "cycle-3/tool"}
        client._request_json(
            "/commands",
            method="POST",
            payload={"action": "grip", "duration": 0.01},
            headers=headers,
        )
        with pytest.raises(RuntimeError, match="HTTP 409"):
            client._request_json(
                "/commands",
                method="POST",
                payload={"action": "release", "duration": 0.01},
                headers=headers,
            )


def test_network_client_bounds_remote_response_wait():
    with NetworkDeviceServer() as server:
        client = NetworkDeviceClient(server.base_url)
        with pytest.raises(TimeoutError, match="did not finish"):
            client.execute(
                7,
                {
                    "action": "release",
                    "duration": 0.2,
                    "response_timeout": 0.01,
                    "operation_key": "cycle-2/release",
                },
            )
        status = client.status()
        assert status["running_operations"] == 0
        assert status["cancelled_operations"] == 1


def test_network_client_preserves_timeout_when_remote_cancel_fails():
    class CancelFailingClient(NetworkDeviceClient):
        def _request_json(self, path, **kwargs):
            if path == "/commands":
                return {"operation_id": "7", "deduplicated": False}
            return {"status": "running"}

        def cancel(self, operation_id):
            raise RuntimeError("cancel endpoint unavailable")

    client = CancelFailingClient("http://127.0.0.1")
    with pytest.raises(TimeoutError, match="operation 7"):
        client.execute(
            7,
            {
                "action": "grip",
                "duration": 0.0,
                "response_timeout": 0.001,
                "operation_key": "cycle-7/grip",
            },
        )


def test_network_client_can_cancel_remote_operation():
    with NetworkDeviceServer() as server:
        client = NetworkDeviceClient(server.base_url)
        submitted = client._request_json(
            "/commands",
            method="POST",
            payload={"action": "grip", "duration": 0.2},
            headers={"Idempotency-Key": "cycle-4/grip"},
        )
        result = client.cancel(submitted["operation_id"])
        assert result == {
            "operation_id": submitted["operation_id"],
            "status": "cancelled",
            "cancelled": True,
        }
        operation = server.device.wait(submitted["operation_id"], timeout=0)
        assert operation["status"] == "cancelled"
        time.sleep(0.25)
        operation = server.device.wait(submitted["operation_id"], timeout=0)
        assert operation["status"] == "cancelled"


def test_network_server_can_stop_before_start_and_rejects_restart():
    server = NetworkDeviceServer()
    server.stop()
    server.stop()
    with pytest.raises(RuntimeError, match="closed"):
        server.start()


def test_network_server_stop_cancels_running_device_operations():
    server = NetworkDeviceServer()
    server.start()
    submitted = server.device.submit(
        {"action": "inspect", "duration": 1.0}, "shutdown/inspection"
    )
    server.stop()
    operation = server.device.wait(submitted["operation_id"], timeout=0)
    assert operation["status"] == "cancelled"
    assert operation["error"] == "device server stopped"
    assert server.device.health()["running_operations"] == 0


def test_network_demo_module_runs_to_completion():
    completed = subprocess.run(
        [sys.executable, "-m", "examples.network_service_demo"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    summary = json.loads(completed.stdout)
    assert summary["result"]["accepted"]


def test_dual_arm_remote_inspection_overlaps_retreat_and_network_wait():
    summary = __import__(
        "examples.dual_arm.05_remote_inspection", fromlist=["run"]
    ).run()
    assert summary["workflow"] == "remote_inspection"
    assert summary["inspection"]["action"] == "inspect"
    assert summary["inspection"]["accepted"]
    assert summary["inspection_completed_before_retreat"]
    assert summary["inspection_timeline"] == ["queued", "running", "succeeded"]
    assert summary["arms"]["left"] == pytest.approx([0.0, 0.25, -0.4])
    assert summary["arms"]["right"] == pytest.approx([0.0, -0.25, 0.4])
