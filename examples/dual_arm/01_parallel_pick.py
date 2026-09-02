"""Move both arms and close both grippers concurrently."""

import json
import time
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
        started = time.monotonic()
        moves = runtime.move_arms(LEFT_PICK, RIGHT_PICK)
        grippers = [
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
        runtime.wait_arms(moves)
        runtime.pilot.wait_requests(grippers, timeout=2.0)
        summary = final_snapshot(runtime)
        summary.update(
            {
                "workflow": "parallel_pick",
                "elapsed": time.monotonic() - started,
                "parallel_moves": moves,
            }
        )
        return summary


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
