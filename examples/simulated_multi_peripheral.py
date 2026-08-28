"""Coordinate camera, vacuum, and PLC services without physical hardware."""

import concurrent.futures
import json
import threading
import time

from orchestrion import (
    CallableTask,
    FieldSpec,
    MoveSyncOption,
    RequestSchema,
    RetryPolicy,
    SimulatedRobotTask,
    TaskPilot,
)
from orchestrion.tasks import PollingTask


def main() -> None:
    state_lock = threading.Lock()
    state = {
        "camera_frames": 0,
        "camera_attempts": 0,
        "vacuum_enabled": False,
        "plc_output": False,
        "camera_observed_at": time.time(),
        "vacuum_observed_at": time.time(),
        "plc_observed_at": time.time(),
        "temperature": 24.0,
    }

    def camera_capture(request_id, content):
        time.sleep(0.01)
        with state_lock:
            state["camera_attempts"] += 1
            if state["camera_attempts"] == 1:
                raise ConnectionError("simulated camera disconnect")
            state["camera_frames"] += 1
            state["camera_observed_at"] = time.time()
            return {"frame_id": state["camera_frames"], "request_id": request_id}

    def vacuum_command(request_id, content):
        enabled = bool((content or {}).get("enabled"))
        with state_lock:
            state["vacuum_enabled"] = enabled
            state["vacuum_observed_at"] = time.time()
        return {"enabled": enabled}

    def plc_write(request_id, content):
        output = bool((content or {}).get("output"))
        with state_lock:
            state["plc_output"] = output
            state["plc_observed_at"] = time.time()
        return {"output": output}

    def read_temperature():
        with state_lock:
            state["temperature"] += 0.1
            return {"celsius": round(state["temperature"], 1)}

    def device_status(kind):
        def snapshot():
            with state_lock:
                details = {
                    "health": "online",
                    "observed_at": state["{}_observed_at".format(kind)],
                    "device_type": kind,
                }
                if kind == "camera":
                    details["frames"] = state["camera_frames"]
                elif kind == "vacuum":
                    details["enabled"] = state["vacuum_enabled"]
                else:
                    details["output"] = state["plc_output"]
                return details

        return snapshot

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    services = {
        "camera": CallableTask(
            camera_capture,
            device_status("camera"),
            retry_policy=RetryPolicy(
                max_attempts=3,
                delay=0.01,
                backoff=2.0,
                retry_exceptions=(ConnectionError,),
            ),
            metadata={"device_type": "camera", "commands": ["capture"]},
            request_schema=RequestSchema(
                {
                    "exposure_ms": FieldSpec(
                        "number", required=True, minimum=0.1, maximum=100.0
                    )
                }
            ),
        ),
        "vacuum": CallableTask(
            vacuum_command,
            device_status("vacuum"),
            metadata={"device_type": "vacuum", "commands": ["set_enabled"]},
            request_schema=RequestSchema(
                {"enabled": FieldSpec("boolean", required=True)}
            ),
        ),
        "plc": CallableTask(
            plc_write,
            device_status("plc"),
            metadata={"device_type": "plc", "commands": ["write_output"]},
            request_schema=RequestSchema(
                {"output": FieldSpec("boolean", required=True)}
            ),
        ),
        "temperature": PollingTask(
            read_temperature,
            interval=0.05,
            metadata={"device_type": "temperature_sensor", "unit": "celsius"},
        ),
    }
    with TaskPilot(
        SimulatedRobotTask([0.0, 0.0]),
        services,
        executor_owned=executor,
    ) as pilot:
        sensor_deadline = time.monotonic() + 1.0
        while not pilot.query_task_status("temperature")["available"]:
            if time.monotonic() >= sensor_deadline:
                raise TimeoutError("temperature sensor did not produce a sample")
            time.sleep(0.005)
        request_ids = [
            pilot.call_srv_async(
                "camera",
                {"exposure_ms": 8},
                MoveSyncOption.no_sync(),
                priority=10,
            ),
            pilot.call_srv_async(
                "vacuum", {"enabled": True}, MoveSyncOption.no_sync()
            ),
            pilot.call_srv_async(
                "plc",
                {"output": True},
                MoveSyncOption.no_sync(),
                idempotency_key="demo-cycle/output-1",
            ),
            pilot.call_srv_async("temperature", sync_option=MoveSyncOption.no_sync()),
        ]
        for result in pilot.wait_requests(request_ids, timeout=1.0):
            print(result)
        print(json.dumps(pilot.describe_services(), indent=2))
        print(json.dumps(pilot.query_health(stale_after=1.0), indent=2))


if __name__ == "__main__":
    main()
