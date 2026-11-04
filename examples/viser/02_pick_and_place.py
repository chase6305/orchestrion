"""Visualize a synchronized pick-and-place sequence."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.viser.common import (
    HOME_Q,
    PAYLOAD_GRASP_POSITION,
    PICK_ABOVE_Q,
    PICK_Q,
    PLACE_ABOVE_Q,
    PLACE_Q,
    ViserDemoRuntime,
    run_demo,
)
from orchestrion import MoveSyncOption


def workflow(runtime: ViserDemoRuntime, args) -> None:
    runtime.add_workspace_markers()
    runtime.set_status("1 / 6 · Approach PICK from above")
    runtime.move_and_wait(PICK_ABOVE_Q, args.steps, args.interval)
    runtime.set_status("2 / 6 · Lower to PICK, then grasp")
    runtime.move(PICK_Q, args.steps, args.interval)
    close_id = runtime.pilot.call_srv_async(
        "gripper",
        {"action": "close", "position": PAYLOAD_GRASP_POSITION},
        MoveSyncOption.sync_w_latest_move(),
    )
    runtime.wait_success(close_id)
    runtime.attach_payload()

    runtime.set_status("3 / 6 · Lift clear of the PICK station")
    runtime.move_and_wait(PICK_ABOVE_Q, args.steps, args.interval)
    runtime.set_status("4 / 6 · Transfer above the stations")
    runtime.move_and_wait(PLACE_ABOVE_Q, args.steps, args.interval)
    runtime.set_status("5 / 6 · Lower to PLACE, then release")
    runtime.move(PLACE_Q, args.steps, args.interval)
    open_id = runtime.pilot.call_srv_async(
        "gripper", {"action": "open"}, MoveSyncOption.sync_w_latest_move()
    )
    runtime.wait_success(open_id)
    runtime.release_payload()
    runtime.set_status("6 / 6 · Retreat above PLACE, then return HOME")
    runtime.move_and_wait(PLACE_ABOVE_Q, args.steps, args.interval)
    runtime.move_and_wait(HOME_Q, args.steps, args.interval)


if __name__ == "__main__":
    run_demo("02 · Pick and Place", workflow)
