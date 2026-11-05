# Changelog

## Unreleased

- Include scheduler tick rounding in default gripper timeouts, so slow animations
  can finish with non-integral sampling intervals; explicit timeouts are unchanged.
- Release cancelled request inputs outside pilot state locks, including during
  shutdown, allowing SDK finalizers to query health from a cleanup thread.
- Skip result scans for service notifications with no active requests, using
  per-service running counts while preserving callback aliases and fallback polls.
- Roll back viewer startup failures, closing the Viser transport and owned
  executor/controller resources when initialization or monitor startup fails.
- Finish dual-arm monitor updates before publishing the final frame, preventing
  late monitor samples or errors from replacing it, even after the join times out.
  Clarify payload-center coordinate names.
- Check native numeric types before the Real ABC during trajectory validation,
  reducing per-joint overhead while preserving finite-float range checks.
- Reuse timing validation in the HTTP demo client, rejecting float overflow and
  underflow before socket operations while accepting representable fractions.
- Clean up gripper request state at future completion, making concurrent queued
  cancellation idempotent and preventing late cancellation flags from accumulating.
- Validate Viser gripper request IDs before sequence reservation or cancellation,
  so boolean/float aliases cannot affect an active integer-ID request.
- Reuse discovered references for workflow inputs without schemas. Derive final
  dependencies and recheck paths after schema hooks replace nested references.
- Fast-path native numeric schema values while preserving exact large integers;
  report overflowing real values as validation errors before command dispatch.
- Reuse each arm's latest rendered pose and kinematics while updating progress,
  grippers and ownership every frame. Serialize complete live/replay frame updates
  and retry pose updates after renderer or kinematics failures.
- Reject underflowing retry settings during policy validation and expose fractional
  retry values as JSON numbers in service descriptions, preserving exact policies.
- Reject positive timing intervals that become zero as native floats, preserving
  motion IDs and avoiding zero-cadence polling. Normalize pilot polling intervals.
- Advertise the actual Viser listening port in browser and SSH instructions when
  the requested port is occupied.
- Locate replay frames with indexed binary search, avoiding a full timestamp
  copy on every seek or playback tick while preserving duplicate-time behavior.
- Capture replay stage and payload ownership from one event snapshot, avoiding
  duplicate history copies and inconsistent frames during concurrent updates.
  Stamp frames under the append lock so overlapping final capture stays ordered.
- Vectorize Viser joint interpolation while preserving endpoint weights and
  independent trajectory points.
- Resolve workflow output references on demand instead of projecting every
  ordering dependency into a temporary output dictionary.
- Release pruned, cancelled threaded responses after SDK cleanup finishes,
  including after pilot shutdown, without changing public response-forgetting rules.
- Reuse terminal counts when pruning request history and retain removed SDK
  payloads until state-lock release, preserving active waiters and newest results.
- Bound slow gripper animation preparation while preserving nominal travel time;
  avoid interval underflow for tiny speeds and stationary commands. Reuse pending
  futures on gripper shutdown instead of inspecting completed response history.
  Include the final sample's hold in estimated default timeouts.
- Correct the Viser pick-and-place payload fit, grasp width and approach/lift
  paths; replace the README recording with the corrected runtime sequence.
- Reduce scheduler and snapshot overhead with linear motion-wait scans, cached
  cancellation history, targeted JSON copies and smaller pending-status allocations.
  Reuse normalized claims and batch durable event-history trimming.
- Reuse owned feedback/checkpoint JSON internally while preserving isolated public
  reads. Check final schema payloads and failure diagnostics for finite JSON,
  isolation and byte limits.
- Release retired SDK values outside locks. Stop failed robot workers, cancel local
  trajectories and preserve cancellation history across explicit restarts.
- Validate copied telemetry, metadata, schema choices and SDK request/motion IDs
  before use. Clarify synchronization, segment endpoints and result retention.
- Share finite timing checks, normalize `Fraction` delays and preserve long wait
  deadlines. Reject unsupported native waits before queue or lifecycle changes.
- Report unbound threaded adapters offline and prevent SDK status fields from
  overriding executor binding.
- Honor gripper cancellation during preparation and move-ID publication; cancel
  accepted dual-arm gripper moves when their waits fail.
- Batch concurrent durable dispatch intents and graph-stop results atomically;
  interrupt remote queries on cancellation and reject inconsistent recovery leases.
- Add optional SQLite workflow journals with committed dispatch intent, durable
  keyed results/checkpoints, conservative hard-exit recovery, exclusive ownership,
  reconciliation evidence and explicit retention. Storage faults close admission.
- Add `TicketAction` and typed asynchronous operation backends; distinguish command
  acceptance, final completion and cancellation confirmation without actuator retries.
- Compose fixed DAGs into resource-scoped capabilities with `workflow_action`,
  hierarchical operation IDs, inner concurrency and versioned execution identity.
- Add action versions, independent stop-confirmation deadlines, draining-worker
  accounting, late ticket checkpoints and opt-in JSON byte limits. Escape operation
  ID components and expose `WorkflowRun.operation_id()` for safe reconciliation.
- Demonstrate durable resource-scoped execution against the loopback HTTP device,
  including restart replay without duplicate commands and lost-response recovery.
- Avoid quadratic choice membership checks during schema construction while
  preserving candidate validation, nested constraints, and defensive copies.
  Add a reproducible scalar/composite enumeration benchmark.
- Validate known input fields and composite choices even when a workflow node
  contains output references; reject invalid plans before any device action and
  recheck resolved values before dispatch. Add binding-independent template validation.
- Distinguish nested JSON booleans from numbers in object/array schema choices.
- Consolidate regression files by behavior, share backend/schema cases through
  parametrization, and replace sleep-based task checks with bounded completion
  signals. Verify queued cancellation actually prevents device dispatch.
- Add nested object/array schemas with element types and length bounds, reuse
  scalar validators, and preserve precise field paths in validation errors.
- Validate declared workflow output schemas before success, reject statically
  impossible output references during compilation, and expose result contracts
  to planners. Invalid vision results cannot dispatch dependent arm commands.
- Add contracts to the simulated multi-device cell and reject future-dated as
  well as stale visual observations before manipulation.
- Bound SDK-thread feedback queues, batch loop notifications, retain recent
  progress on overflow, and expose dropped counts in node results and diagnostics.
- Supervise unexpected scheduler exits, block queued adapters before device I/O,
  release unstarted reservations, and drain active operations independently of the
  scheduler task while preserving uncertain resource ownership.
- Deduplicate explicitly keyed workflow submissions against their compiled input
  snapshots; retain original outcomes with bounded key admission and explicit
  pruning that protects active and unreconciled operations.
- Explain dependency, concurrency, resource queue and stop-confirmation waits;
  expose scheduler counts, resource owners and submission retention diagnostics.
- Add whole-workflow execution budgets covering dependency and resource waits;
  preserve device claims during timeout cleanup and stop fail-fast admission as
  soon as a node deadline expires.
- Prevent late cancellation from rewriting completed workflow success, suppress
  cancelled SDK calls still waiting in an executor queue, and reject action waits
  or shutdowns that would wait on their own engine.
- Recover workflow resource accounting after task startup failures or worker
  interruption, isolate malformed SDK exception messages, support thread-safe
  feedback, and consume late executor failures after an interrupted await.
- Add experimental `orchestrion.workflow` for heterogeneous DAG execution, planner
  proposals, output bindings, mandatory action resource claims, capacity scheduling,
  cooperative cancellation, and quarantine/reconciliation of uncertain operations.
- Add a simulated arm, gripper, fixture, speech, vision, and mobile-base workflow,
  plus an architecture and migration design separating planning from execution.
- Apply priorities consistently across immediate and ready motion-synchronized
  requests, collect inline results before eviction, and avoid repeated full polling
  while admitting a backlog. Idle schedulers now wait for notifications.
- Isolate malformed, missing, mismatched, and unreadable device responses so an
  adapter failure does not terminate the scheduler or leave a request running forever.
- Fix cancellation/pruning races and delegate cancellation outside the shared
  state lock while preserving completions that arrive during cancellation.
- Retain every result needed by a grouped wait, validate all IDs before waiting,
  and maintain request status counts without scanning the full history.
- Reject overlapping scheduler/polling restarts and requests prepared across a
  stop/restart boundary; allow callbacks to stop their own pilot or robot without
  joining themselves or deadlocking with an external shutdown.
- Allocate robot move IDs atomically with queue admission, reject malformed
  trajectories, and preserve completed IDs across modular-robot restarts.
- Add main-move cancellation history to `ReducedRobotState` and propagate stopped
  main movements to synchronized service requests. Preserve submodule cancellation
  outcomes when later movements finish.
- Remove quadratic sparse-ID history copying and pending-future scans from thread
  pool submission and bounded response retention; add reproducible microbenchmarks.
- Retain inline failures and report concurrent inline calls as pending until done.
- Stop zero-delay retries and prevent old invocations from resuming retries after
  restart; preserve exponential-delay behavior at extreme numeric ranges.
- Keep integer schema bounds exact and snapshot field constraints at construction.
- Validate malformed network commands and body lengths, close command admission
  before server shutdown, release sockets on startup failure, and return defensive
  operation snapshots.
- Preserve main motion queued before simulated-robot initialization, roll back
  failed worker starts, and use constant-time deque removal for submodule queues.
- Keep health snapshots JSON-compatible when one adapter returns invalid metadata.
- Test installations with and without Viser, fail on unhandled background-thread
  exceptions, and smoke-test built wheels in isolated CI environments.

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
