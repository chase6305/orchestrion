"""Visualize a smooth UR5 joint-space sweep and return home."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.viser.common import HOME_Q, PICK_Q, ViserDemoRuntime, run_demo


def workflow(runtime: ViserDemoRuntime, args) -> None:
    runtime.server.gui.add_markdown("Smooth joint interpolation: **HOME → PICK → HOME**")
    runtime.set_status("moving HOME → PICK")
    runtime.move_and_wait(PICK_Q, args.steps, args.interval)
    runtime.set_status("moving PICK → HOME")
    runtime.move_and_wait(HOME_Q, args.steps, args.interval)


if __name__ == "__main__":
    run_demo("01 · Joint Sweep", workflow)
