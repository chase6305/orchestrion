"""Show two peripheral actions synchronized to trajectory segment endpoints."""


from examples.viser.common import (
    HOME_Q,
    PICK_Q,
    PLACE_Q,
    ViserDemoRuntime,
    interpolate,
    run_demo,
)
from orchestrion import MoveSyncOption


def workflow(runtime: ViserDemoRuntime, args) -> None:
    runtime.add_workspace_markers(include_payload=False)
    first = interpolate(HOME_Q, PICK_Q, args.steps)
    second = interpolate(PICK_Q, PLACE_Q, args.steps)[1:]
    trajectory = first + second
    endpoint_index = [len(first), len(trajectory)]
    move_begin, move_end = runtime.pilot.move_joint_trajectory_async(
        trajectory,
        interval=args.interval,
        endpoint_index=endpoint_index,
    )
    if move_begin < 0:
        raise RuntimeError("Segmented trajectory was rejected")
    runtime.set_status("segment 1: move to PICK and close gripper")
    close_id = runtime.pilot.call_srv_async(
        "gripper",
        {"action": "close"},
        MoveSyncOption.sync_w_explicit_id(move_begin),
    )
    open_id = runtime.pilot.call_srv_async(
        "gripper",
        {"action": "open"},
        MoveSyncOption.sync_w_explicit_id(move_end - 1),
    )
    runtime.server.scene.add_label(
        "/world/segment_info",
        "green endpoint: close\nblue endpoint: open",
        position=(0.0, 0.0, 0.8),
    )
    runtime.wait_success(close_id)
    runtime.set_status("segment 2: move to PLACE and open gripper")
    runtime.wait_success(open_id)
    runtime.wait_motion()


if __name__ == "__main__":
    run_demo("03 · Segmented Synchronization", workflow)
