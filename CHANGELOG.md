# Changelog

## 0.2.0

- Add unified request status, results, waiting, cancellation, and timelines.
- Add event-driven scheduling with compatibility polling.
- Add `SimulatedRobotTask` and a hardware-free pick-and-place example.
- Make function-call result storage and cancellation thread-safe.
- Add Python 3.9/3.12 CI, Ruff checks, and expanded lifecycle tests.
- Add submodule movement IDs, state queries, completion waits, and cancellation.
- Make the Viser gripper plan from its current position and report real completion.
- Preserve gripper command order under concurrent execution and validate commands
  before motion is queued.
- Add FK-aligned pick/place markers and payload attachment/release visualization.
- Add eight focused Viser demos, including interactive controls, timelines,
  motion trails, and a gripper cancellation/reversal lab.
- Roll back partially initialized task sets and reject non-finite timing inputs.
- Reject malformed trajectory endpoint lists instead of crashing scheduler threads.
- Keep `TaskPilot.stop()` bounded when running callbacks ignore cancellation.
- Add safe pruning for terminal request history and retained task responses.
- Snapshot queued request content and harden task stop/restart behavior.
- Reject overlapping modular-robot DOF ranges and invalid wait timing values.
- Honor main-trajectory intervals and wake movement waiters on state changes.
- Make modular/Viser synchronization event-driven and isolate marker FK state.
- Reject non-finite or non-numeric initial joints and trajectory values.
- Make initialization rollback transactional across robot, tasks, callbacks, and threads.
- Clear pending robot trajectories on stop so restart cannot execute stale motion.
- Isolate failing state/completion observers from scheduler worker threads.
- Ensure callback cleanup failures cannot skip task or robot shutdown.
- Make Viser demos wait for a browser, remain open by default, and report phases.
- Add a consistent camera/world frame and replace the abstract trail with EE FK.
- Hide payload props from demos that do not perform grasping.
