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
- Add direct-script support and a unified `python -m examples.viser` launcher.
- Add TaskPilot peripheral-status APIs and live gripper health in the Viser GUI.
- Isolate peripheral monitoring failures so one offline device cannot stop a snapshot.
- Add a JSON-compatible system health snapshot for dashboards and health endpoints.
- Add context-managed lifecycle helpers and protect the service registry from mutation.
- Normalize device health/freshness and add a callable adapter plus multi-device example.
- Add revision-based health change waiting for event-driven monitoring clients.
- Add a generated multi-device orchestration illustration to the README.
- Add a reference-style architecture infographic and refresh the Mermaid source map.
- Add opt-in bounded retry/backoff for idempotent callable peripheral services.
- Add `PollingTask` for cached continuous telemetry and sensor health reporting.
- Add shared-deadline multi-request waiting and service-filtered bulk cancellation.
- Serialize concurrent TaskPilot initialize/stop lifecycle calls.
- Add stable integer priorities for queued and motion-synchronized requests.
- Add service-scoped idempotency keys for safe command retransmission.
- Add side-effect-free service discovery with capabilities and adapter metadata.
- Add declarative command schemas with validation before device I/O and discovery metadata.
- Refresh the Viser presentation with a dark control panel and real runtime capture.
- Cap Viser gripper rendering at 60 Hz so fast headless demos remain reliable.
- Expose Viser gripper commands, units, and position range through service discovery.
- Replace the static Viser preview with a compact recording of the real demo 02 workflow.
- Add a ticket-based HTTP device demo with long-poll response waiting, health
  discovery, transport retries, and end-to-end idempotency.
- Harden the network demo with server-side command validation, HTTP 409
  idempotency conflicts, bounded client settings, and remote timeout cancellation.
- Add a reusable dual-arm simulation runtime, three collaboration demos, shared
  workspace reservation service, and shared-deadline multi-submodule waiting.
- Add coordinated dual-arm abort behavior so a cancelled arm stops its moving peer.
- Add a dual-arm remote-inspection demo that overlaps arm retreat with a ticketed
  HTTP quality-service request.
- Add TaskPilot service synchronization against individual submodule move IDs,
  including cancellation propagation from the associated movement.
- Add an event-driven dual-arm health monitor demo based on monotonic revisions.
- Let all six dual-arm workflows run with `--viser`, using two live UR5 models
  and a shared-zone, gripper, and orchestration-event status panel.
- Refine dual-arm Viser poses with mirrored UR5 motion anchors, presentation
  pacing, live workflow phases, and an animated handoff payload.
- Record every dual-arm Viser workflow for offline replay with play, pause,
  restart, timeline scrubbing, speed selection, auto-play, and looping.
- Save and load versioned dual-arm replay JSON files without initializing robot,
  peripheral, or network services during playback.
- Add a complete dual-arm handoff-and-place production cycle and enrich the Viser
  cell with robot pedestals, labeled input/output stations, and output payload state.
- Replace hand-positioned dual-arm props with FK-derived input, handoff, and output
  locations, using UR5 pose anchors whose handoff endpoints agree within 2 mm.
- Constrain UR5 handoff orientation as well as position and replace symbolic
  end-effector blocks with animated two-finger grippers around the payload axis.
