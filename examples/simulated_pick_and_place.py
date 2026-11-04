"""Move to PICK, close the gripper, carry to PLACE, then release.

Run from the repository root: python -m examples.simulated_pick_and_place
Only the Python standard library and Orchestrion are required.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from orchestrion import (
    CallableTask,
    MoveSyncOption,
    RequestStatus,
    SimulatedRobotTask,
    TaskPilot,
)


def run_demo() -> Dict[str, Any]:
    """Return command observations and request histories for one simulated cycle."""
    robot = SimulatedRobotTask([0.0, 0.0])

    def grip(request_id, command):
        # A real adapter must return only after the gripper finishes its action.
        return {"gripper": command["action"], "joints": robot.query_state().jps}

    steps = []
    with TaskPilot(
        robot,
        {"gripper": CallableTask(grip)},
        executor_owned=ThreadPoolExecutor(max_workers=1),
    ) as pilot:
        for station, path, action in (
            ("PICK", [[0.2, 0.1], [0.4, 0.2]], "close"),
            ("PLACE", [[0.1, 0.4], [-0.3, 0.5]], "open"),
        ):
            move_begin, _ = pilot.move_joint_trajectory_async(path, interval=0.05)
            if move_begin < 0:
                raise RuntimeError("trajectory rejected at " + station)
            request_id = pilot.call_srv_async(
                "gripper", {"action": action}, MoveSyncOption.sync_w_latest_move()
            )
            result = pilot.wait_request(request_id, timeout=2.0)
            if result.status is not RequestStatus.SUCCEEDED:
                raise RuntimeError(
                    "gripper failed at {}: {}".format(station, result.error)
                )
            steps.append({
                "station": station,
                **result.content,
                "timeline": [
                    event.status.value for event in pilot.timeline(request_id)
                ],
            })
    return {"status": "succeeded", "steps": steps}


def main() -> None:
    summary = run_demo()
    for step in summary["steps"]:
        print("{}: {} at {}".format(step["station"], step["gripper"], step["joints"]))
    print("Cycle: " + summary["status"])


if __name__ == "__main__":
    main()
