"""Visualize a synchronized pick-and-place sequence."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.viser.common import HOME_Q, PICK_Q, PLACE_Q, ViserDemoRuntime, run_demo
from orchestrion import MoveSyncOption


def workflow(runtime: ViserDemoRuntime, args) -> None:
    runtime.add_workspace_markers()
    runtime.set_status("1 / 4 · Move to PICK, then close")
    runtime.move(PICK_Q, args.steps, args.interval)
    close_id = runtime.pilot.call_srv_async(
        "gripper", {"action": "close"}, MoveSyncOption.sync_w_latest_move()
    )
    runtime.wait_success(close_id)
    runtime.attach_payload()

    runtime.set_status("2 / 4 · Lift with gripper closed")
    runtime.move_and_wait(HOME_Q, args.steps, args.interval)
    runtime.set_status("3 / 4 · Move to PLACE, then open")
    runtime.move(PLACE_Q, args.steps, args.interval)
    open_id = runtime.pilot.call_srv_async(
        "gripper", {"action": "open"}, MoveSyncOption.sync_w_latest_move()
    )
    runtime.wait_success(open_id)
    runtime.release_payload()
    runtime.set_status("4 / 4 · Payload placed; return HOME")
    runtime.move_and_wait(HOME_Q, args.steps, args.interval)


if __name__ == "__main__":
    run_demo("02 · Pick and Place", workflow)
