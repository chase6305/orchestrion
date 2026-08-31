import json
import subprocess
import sys

import pytest

from examples.network_device_service import NetworkDeviceServer, SimulatedNetworkDevice
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
