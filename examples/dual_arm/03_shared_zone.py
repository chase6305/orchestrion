"""Demonstrate shared-workspace exclusion and recovery after rejection."""

import json
from typing import Callable, Dict

from examples.dual_arm.common import DualArmRuntime, final_snapshot
from orchestrion import RequestStatus


def run(
    runtime_factory: Callable[[], DualArmRuntime] = DualArmRuntime,
) -> Dict:
    with runtime_factory() as runtime:
        first = runtime.command_zone("reserve", "cycle-a")
        first_result = runtime.pilot.wait_request(first, timeout=1.0)
        blocked = runtime.command_zone("reserve", "cycle-b")
        blocked_result = runtime.pilot.wait_request(blocked, timeout=1.0)
        if blocked_result.status is not RequestStatus.FAILED:
            raise AssertionError("conflicting shared-zone reservation was accepted")

        release_a = runtime.command_zone("release", "cycle-a")
        runtime.pilot.wait_request(release_a, timeout=1.0)
        retry = runtime.command_zone("reserve", "cycle-b")
        retry_result = runtime.pilot.wait_request(retry, timeout=1.0)
        release_b = runtime.command_zone("release", "cycle-b")
        runtime.pilot.wait_request(release_b, timeout=1.0)

        summary = final_snapshot(runtime)
        summary.update(
            {
                "workflow": "shared_zone",
                "first_status": first_result.status.value,
                "blocked_status": blocked_result.status.value,
                "blocked_error": blocked_result.error,
                "retry_status": retry_result.status.value,
            }
        )
        return summary


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
