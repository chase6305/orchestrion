"""Render the end-effector Cartesian trail while animating the robot."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from examples.viser.common import (
    HOME_Q,
    PICK_Q,
    PLACE_Q,
    ViserDemoRuntime,
    interpolate,
    run_demo,
)


def workflow(runtime: ViserDemoRuntime, args) -> None:
    runtime.add_workspace_markers(include_payload=False)
    poses = [HOME_Q, PICK_Q, PLACE_Q, HOME_Q]
    colors = [(230, 80, 80), (80, 200, 100), (70, 120, 240)]
    for index, (start, end) in enumerate(zip(poses, poses[1:])):
        trajectory = interpolate(start, end, args.steps)
        points = np.asarray(
            [runtime.fk_position(np.asarray(joints)) for joints in trajectory]
        )
        segments = np.stack([points[:-1], points[1:]], axis=1)
        runtime.server.scene.add_line_segments(
            "/world/joint_trail/segment_{}".format(index),
            points=segments,
            colors=colors[index],
            line_width=3.0,
        )
        move_begin, _ = runtime.pilot.move_joint_trajectory_async(
            trajectory, interval=args.interval
        )
        if move_begin < 0:
            raise RuntimeError("Trail trajectory {} was rejected".format(index))
        runtime.set_status("playing trail segment {}/3".format(index + 1))
        runtime.wait_motion()


if __name__ == "__main__":
    run_demo("07 · End-Effector Motion Trail", workflow)
