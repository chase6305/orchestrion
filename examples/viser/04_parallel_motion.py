"""Animate arm and gripper concurrently without move synchronization."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.viser.common import PICK_Q, ViserDemoRuntime, run_demo
from orchestrion import MoveSyncOption


def workflow(runtime: ViserDemoRuntime, args) -> None:
    runtime.server.gui.add_markdown(
        "The arm moves while the gripper closes immediately (`no_sync`)."
    )
    runtime.set_status("moving arm and gripper in parallel")
    runtime.move(PICK_Q, args.steps, args.interval)
    request_id = runtime.pilot.call_srv_async(
        "gripper", {"action": "close"}, MoveSyncOption.no_sync()
    )
    runtime.wait_success(request_id)
    runtime.wait_motion()


if __name__ == "__main__":
    run_demo("04 · Parallel Arm and Gripper", workflow)
