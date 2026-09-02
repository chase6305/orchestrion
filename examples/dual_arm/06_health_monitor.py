"""Observe a dual-arm cycle through event-driven health revisions."""

import json
from typing import Callable, Dict

from examples.dual_arm.common import (
    LEFT_PICK,
    RIGHT_PICK,
    DualArmRuntime,
    final_snapshot,
)
from orchestrion import MoveSyncOption


def run(
    runtime_factory: Callable[[], DualArmRuntime] = DualArmRuntime,
) -> Dict:
    with runtime_factory() as runtime:
        moves = runtime.move_arms(LEFT_PICK, RIGHT_PICK, steps=20, interval=0.004)
        requests = [
            runtime.command_gripper(
                "left",
                "close",
                MoveSyncOption.sync_w_submodule("left_arm", moves["left_arm"]),
            ),
            runtime.command_gripper(
                "right",
                "close",
                MoveSyncOption.sync_w_submodule("right_arm", moves["right_arm"]),
            ),
        ]
        revision = runtime.pilot.health_revision
        revisions = []
        request_samples = []
        while not all(
            runtime.pilot.query_request(request_id).status.terminal
            for request_id in requests
        ):
            snapshot = runtime.pilot.wait_health_change(revision, timeout=1.0)
            if snapshot is None:
                raise TimeoutError("dual-arm health stream stopped changing")
            revision = snapshot["revision"]
            revisions.append(revision)
            if not request_samples or request_samples[-1] != snapshot["requests"]:
                request_samples.append(snapshot["requests"])
        runtime.wait_arms(moves)
        runtime.pilot.wait_requests(requests, timeout=1.0)
        summary = final_snapshot(runtime)
        summary.update(
            {
                "workflow": "health_monitor",
                "observed_revisions": revisions,
                "request_samples": request_samples,
            }
        )
        return summary


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
