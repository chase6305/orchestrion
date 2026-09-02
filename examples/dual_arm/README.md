# Dual-arm demos

These hardware-free examples model a torso, two independent three-joint arms,
two grippers, and a shared handoff zone. They exercise the same asynchronous
submodule and service APIs that a real dual-arm backend would implement.

```bash
python -m examples.dual_arm --list
python -m examples.dual_arm 01
python -m examples.dual_arm 02
python -m examples.dual_arm 03
python -m examples.dual_arm 04
python -m examples.dual_arm 05
python -m examples.dual_arm 06
python -m examples.dual_arm 07
```

Install the optional visualization dependencies and add `--viser` to render any
of the six workflows with two live UR5 models:

```bash
pip install -e ".[viser]"
python -m examples.dual_arm 01 --viser
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

The browser scene keeps the scheduler's abstract three-axis poses while mapping
them onto FK-validated six-axis UR5 pose anchors. Input, handoff, and output
objects are placed from the resulting end-effector transforms instead of manual
scene coordinates; the two handoff end effectors meet within 2 mm, share the same
payload axis, and face one another across the grasp. Animated two-finger grippers
close around the FK-transformed payload center. This visual adapter does not
change workflow or synchronization semantics. The
panel reports both arm states, recent orchestration events, gripper state, and
the current shared-zone owner. Green/orange end-effector blocks show open/closed
grippers, while the shared zone turns red whenever a cycle reserves it. Visual
mode also uses presentation-paced trajectories and shows the payload moving from
the table to the left arm, through the overlap grasp, and finally to the right arm.

| Demo | Focus |
| --- | --- |
| 01 Parallel Pick | Independent left/right trajectories and gripper services executing concurrently |
| 02 Payload Handoff | Pick, shared-zone reservation, coordinated rendezvous, overlap grasp, and ownership transfer |
| 03 Shared Zone Safety | Conflicting reservation rejection, release, and successful retry |
| 04 Coordinated Abort | One arm is cancelled and the still-moving peer is stopped as a group-safety response |
| 05 Remote Inspection | Arms retreat in parallel while a ticket-based HTTP quality service completes |
| 06 Health Revision Monitor | Event-driven request and health revisions during synchronized arm/gripper work |
| 07 Handoff and Place | Complete input pick, protected handoff, output placement, and return-home production cycle |

`common.py` contains the reusable pieces:

- `DualArmRobot` maps the torso, arms, and grippers onto independent modular
  submodules.
- `DualArmRuntime` owns the robot, `TaskPilot`, thread pool, gripper command
  services, and structured event log.
- `SharedZoneCoordinator` prevents unrelated cycles from entering the handoff
  workspace simultaneously while allowing idempotent reservations by the same
  cycle.
- `wait_submodule_moves()` waits for both arms against one shared deadline and
  fails the group if either arm is cancelled.
- `move_submodules_trajectories_async()` atomically validates and queues the two
  trajectories, while `query_submodule_states()` returns a coherent joint snapshot.
- `cancel_submodule_moves()` stops the coordinated group under one state lock when
  either arm fails or the shared deadline expires.
- `MoveSyncOption.sync_w_submodule()` lets each gripper request enter
  `WAITING_FOR_MOVE` until its corresponding arm reaches the commanded pose.

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
