"""Retreat both arms while a ticket-based HTTP quality check is running."""

import json
from typing import Callable, Dict

from examples.dual_arm.common import (
    LEFT_HANDOFF,
    LEFT_HOME,
    RIGHT_HANDOFF,
    RIGHT_HOME,
    DualArmRuntime,
    final_snapshot,
)
from examples.network_device_service import NetworkDeviceServer
from examples.network_service_demo import NetworkDeviceClient
from orchestrion import CallableTask, FieldSpec, MoveSyncOption, RequestSchema


def run(
    runtime_factory: Callable[..., DualArmRuntime] = DualArmRuntime,
) -> Dict:
    with NetworkDeviceServer() as server:
        client = NetworkDeviceClient(server.base_url)
        inspection = CallableTask(
            client.execute,
            status=client.status,
            request_schema=RequestSchema(
                {
                    "action": FieldSpec(
                        "string", required=True, choices=("inspect",)
                    ),
                    "duration": FieldSpec(
                        "number", required=True, minimum=0.0, maximum=2.0
                    ),
                    "response_timeout": FieldSpec(
                        "number", required=True, minimum=0.01, maximum=10.0
                    ),
                    "operation_key": FieldSpec("string", required=True),
                }
            ),
            metadata={"transport": "http", "role": "quality_gate"},
        )
        with runtime_factory(extra_services={"inspection": inspection}) as runtime:
            approach = runtime.move_arms(LEFT_HANDOFF, RIGHT_HANDOFF)
            runtime.wait_arms(approach)
            inspection_id = runtime.pilot.call_srv_async(
                "inspection",
                {
                    "action": "inspect",
                    "duration": 0.02,
                    "response_timeout": 1.0,
                    "operation_key": "part-42/inspection",
                },
                MoveSyncOption.no_sync(),
                idempotency_key="part-42/inspection",
            )
            runtime.record("inspection_requested", request_id=inspection_id)
            retreat = runtime.move_arms(
                LEFT_HOME, RIGHT_HOME, steps=30, interval=0.005
            )
            runtime.wait_arms(retreat)
            retreat_finished_at = next(
                event["timestamp"]
                for event in reversed(runtime.events)
                if event["event"] == "arms_finished"
            )
            result = runtime.pilot.wait_request(inspection_id, timeout=2.0)
            runtime.record(
                "inspection_finished",
                request_id=inspection_id,
                operation_id=result.content["operation_id"],
            )
            summary = final_snapshot(runtime)
            summary.update(
                {
                    "workflow": "remote_inspection",
                    "inspection": result.content,
                    "inspection_timeline": [
                        event.status.value
                        for event in runtime.pilot.timeline(inspection_id)
                    ],
                    "inspection_completed_before_retreat": (
                        result.content["remote_completed_at"] <= retreat_finished_at
                    ),
                    "retreat_moves": retreat,
                }
            )
            return summary


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
