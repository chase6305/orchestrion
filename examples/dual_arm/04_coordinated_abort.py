"""Abort the peer arm when one member of a coordinated move is cancelled."""

import json
import time
from typing import Callable, Dict

from examples.dual_arm.common import (
    LEFT_HANDOFF,
    RIGHT_HANDOFF,
    DualArmRuntime,
    final_snapshot,
)


def run(
    runtime_factory: Callable[[], DualArmRuntime] = DualArmRuntime,
) -> Dict:
    with runtime_factory() as runtime:
        moves = runtime.move_arms(
            LEFT_HANDOFF, RIGHT_HANDOFF, steps=50, interval=0.01
        )
        time.sleep(0.02)
        runtime.robot.cancel_submodule_move("right_arm", moves["right_arm"])
        error = None
        try:
            runtime.wait_arms(moves, timeout=1.0)
        except RuntimeError as exc:
            error = str(exc)
        if error is None:
            raise AssertionError("coordinated cancellation did not fail the group")
        summary = final_snapshot(runtime)
        summary.update(
            {
                "workflow": "coordinated_abort",
                "error": error,
                "left_cancelled": moves["left_arm"]
                in runtime.robot.query_submodule_state(
                    "left_arm"
                ).cancelled_move_ids,
                "right_cancelled": moves["right_arm"]
                in runtime.robot.query_submodule_state(
                    "right_arm"
                ).cancelled_move_ids,
            }
        )
        return summary


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
