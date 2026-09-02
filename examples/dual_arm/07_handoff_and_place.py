"""Run a complete pick, handoff, place, and home dual-arm production cycle."""

import json
from typing import Callable, Dict

from examples.dual_arm.common import (
    LEFT_HANDOFF,
    LEFT_HOME,
    LEFT_PICK,
    RIGHT_HANDOFF,
    RIGHT_HOME,
    RIGHT_PICK,
    DualArmRuntime,
    final_snapshot,
)


def run(
    runtime_factory: Callable[[], DualArmRuntime] = DualArmRuntime,
) -> Dict:
    cycle = "production-cycle-1"
    with runtime_factory() as runtime:
        approach = runtime.move_arms(LEFT_PICK, RIGHT_HOME)
        runtime.wait_arms(approach)
        request = runtime.command_gripper("left", "close")
        runtime.pilot.wait_request(request, timeout=2.0)
        runtime.record("payload_owner_changed", owner="left_arm")

        reservation = runtime.command_zone("reserve", cycle)
        runtime.pilot.wait_request(reservation, timeout=2.0)
        rendezvous = runtime.move_arms(LEFT_HANDOFF, RIGHT_HANDOFF)
        runtime.wait_arms(rendezvous)
        request = runtime.command_gripper("right", "close")
        runtime.pilot.wait_request(request, timeout=2.0)
        runtime.record("payload_owner_changed", owner="both_arms")
        request = runtime.command_gripper("left", "open")
        runtime.pilot.wait_request(request, timeout=2.0)
        runtime.record("payload_owner_changed", owner="right_arm")
        release = runtime.command_zone("release", cycle)
        runtime.pilot.wait_request(release, timeout=2.0)

        place = runtime.move_arms(LEFT_HOME, RIGHT_PICK)
        runtime.wait_arms(place)
        request = runtime.command_gripper("right", "open")
        runtime.pilot.wait_request(request, timeout=2.0)
        runtime.record("payload_owner_changed", owner="output_table")

        home = runtime.move_arms(LEFT_HOME, RIGHT_HOME)
        runtime.wait_arms(home)
        summary = final_snapshot(runtime)
        summary.update(
            {
                "workflow": "handoff_and_place",
                "payload_owner": "output_table",
                "rendezvous_moves": rendezvous,
                "place_moves": place,
                "home_moves": home,
            }
        )
        return summary


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
