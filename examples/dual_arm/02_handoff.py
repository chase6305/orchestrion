"""Transfer one payload from the left arm to the right arm."""

import json
from typing import Callable, Dict

from examples.dual_arm.common import (
    LEFT_HANDOFF,
    LEFT_PICK,
    RIGHT_HANDOFF,
    RIGHT_HOME,
    DualArmRuntime,
    final_snapshot,
)


def run(
    runtime_factory: Callable[[], DualArmRuntime] = DualArmRuntime,
) -> Dict:
    cycle = "handoff-1"
    owner = "table"
    with runtime_factory() as runtime:
        approach = runtime.move_arms(LEFT_PICK, RIGHT_HOME)
        runtime.wait_arms(approach)
        left_close = runtime.command_gripper("left", "close")
        runtime.pilot.wait_request(left_close, timeout=1.0)
        owner = "left_arm"
        runtime.record("payload_owner_changed", owner=owner)

        reservation = runtime.command_zone("reserve", cycle)
        runtime.pilot.wait_request(reservation, timeout=1.0)
        handoff = runtime.move_arms(LEFT_HANDOFF, RIGHT_HANDOFF)
        runtime.wait_arms(handoff)

        right_close = runtime.command_gripper("right", "close")
        runtime.pilot.wait_request(right_close, timeout=1.0)
        owner = "both_arms"
        runtime.record("payload_owner_changed", owner=owner)
        left_open = runtime.command_gripper("left", "open")
        runtime.pilot.wait_request(left_open, timeout=1.0)
        owner = "right_arm"
        runtime.record("payload_owner_changed", owner=owner)

        release = runtime.command_zone("release", cycle)
        runtime.pilot.wait_request(release, timeout=1.0)
        summary = final_snapshot(runtime)
        summary.update(
            {
                "workflow": "handoff",
                "payload_owner": owner,
                "handoff_moves": handoff,
            }
        )
        return summary


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
