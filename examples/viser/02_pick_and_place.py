"""Visualize a synchronized pick-and-place sequence."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.viser.common import HOME_Q, PICK_Q, PLACE_Q, ViserDemoRuntime, run_demo
from orchestrion import MoveSyncOption


def gripper_after(runtime, action: str, move_end: int) -> int:
    return runtime.pilot.call_srv_async(
        "gripper",
        {"action": action},
        MoveSyncOption.sync_w_explicit_id(move_end - 1),
    )


def workflow(runtime: ViserDemoRuntime, args) -> None:
    runtime.add_workspace_markers()
    runtime.set_status("approaching PICK")
    _, pick_end = runtime.move(PICK_Q, args.steps, args.interval)
    close_id = gripper_after(runtime, "close", pick_end)
    runtime.wait_success(close_id)
    runtime.attach_payload()

    runtime.set_status("lifting payload")
    runtime.move_and_wait(HOME_Q, args.steps, args.interval)
    runtime.set_status("carrying payload to PLACE")
    _, place_end = runtime.move(PLACE_Q, args.steps, args.interval)
    open_id = gripper_after(runtime, "open", place_end)
    runtime.wait_success(open_id)
    runtime.release_payload()
    runtime.set_status("returning HOME")
    runtime.move_and_wait(HOME_Q, args.steps, args.interval)


if __name__ == "__main__":
    run_demo("02 · Pick and Place", workflow)
