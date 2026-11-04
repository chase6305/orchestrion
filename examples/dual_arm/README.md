# Dual-arm demos

These hardware-free examples model a torso, two independent three-joint arms,
two grippers, and a shared handoff zone. They exercise the same asynchronous
submodule and service APIs that a real dual-arm backend would implement.

```bash
python -m examples.dual_arm --list
python -m examples.dual_arm 02
```

Install the optional visualization dependencies and add `--viser` to render any
of the seven workflows with two live UR5 models:

```bash
python -m pip install -e ".[viser]"
python -m examples.dual_arm 02 --viser --port 8081
python -m examples.dual_arm 05 --viser --no-wait-for-client --duration 3
```

Every visual workflow is recorded at runtime. After completion, use the Viser
panel to play, pause, restart, scrub the timeline, select `0.25x`–`3x` speed, or
enable looping. Automatic replay is available from the command line:

```bash
python -m examples.dual_arm 02 --viser --replay
python -m examples.dual_arm 02 --viser --loop-replay --replay-speed 0.5
```

Replay is offline: it renders captured robot and orchestration state without
calling services again, allocating new request IDs, or repeating network work.
Recordings can also be saved and replayed in a later process:

```bash
python -m examples.dual_arm 02 --viser --recording-out handoff.json
python -m examples.dual_arm 02 --viser --replay-file handoff.json
python -m examples.dual_arm 02 --viser --replay-file handoff.json --loop-replay
```

The versioned JSON file contains visual state only, so it is portable and safe to
share. Loading it does not initialize `TaskPilot` or any device/network service.

The browser shows both UR5 arms, animated grippers, payload ownership, the shared
zone and recent events. It maps the simulated joint paths onto predefined UR5
poses; this is a visualization of the workflow, not a motion planner.

| Demo | Focus |
| --- | --- |
| 01 Parallel Pick | Independent left/right trajectories and gripper services executing concurrently |
| 02 Payload Handoff | Pick, shared-zone reservation, coordinated rendezvous, overlap grasp, and ownership transfer |
| 03 Shared Zone Safety | Conflicting reservation rejection, release, and successful retry |
| 04 Coordinated Abort | One arm is cancelled and the still-moving peer is stopped as a group-safety response |
| 05 Remote Inspection | Arms retreat in parallel while a ticket-based HTTP quality service completes |
| 06 Health Revision Monitor | Event-driven request and health revisions during synchronized arm/gripper work |
| 07 Handoff and Place | Complete input pick, protected handoff, output placement, and return-home production cycle |

[`common.py`](common.py) owns the simulated robot, service adapters, shared-zone
coordinator and event log. Arm moves use atomic batch submission and a shared
wait deadline; synchronized gripper requests depend on their own arm's move ID.

The poses are deliberately abstract joint vectors, so the default examples run
without URDFs or visualization dependencies. `viser.py` is an optional observer
and runtime adapter; a hardware integration can retain the workflow and replace
`DualArmRobot` with a backend that forwards trajectories to the real controllers.

## Service synchronization

The runtime exposes `left_gripper`, `right_gripper`, and `shared_zone` through one
`TaskPilot`. A gripper command can use
`MoveSyncOption.sync_w_submodule("left_arm", move_id)` instead of manually
blocking the application thread. Its request lifecycle then records
`QUEUED -> WAITING_FOR_MOVE -> RUNNING -> SUCCEEDED`. If that arm move is
cancelled, the waiting gripper request becomes `CANCELLED` and never reaches the
device adapter.

The remote-inspection demo adds an `inspection` service at runtime. This shows how
cell-specific HTTP, PLC, vision, or quality services can be composed with the same
dual-arm runtime without modifying its scheduler. Its summary compares the remote
completion timestamp with the arm-retreat event to make the overlap observable.
