# Test organization

Tests are grouped by behavior. Add regressions to the file that owns the contract;
use a new file when a distinct subsystem needs its own setup and assertions.
Keep integration scenarios alongside the corresponding example or public API.

| File | Responsibility |
| --- | --- |
| `test_task.py` | Inline and threaded calls, retries, telemetry, result retention, task lifecycle |
| `test_task_pilot.py` | Motion synchronization, dispatch, scheduler lifecycle and cancellation races |
| `test_legacy_api.py` | Public request, health, lifecycle and simulated-motion compatibility |
| `test_schema.py` | Flat/nested input validation, templates, composite choices, defensive copies and reference paths |
| `test_robot_task.py` | Shared simulated/modular robot contracts and lifecycle races |
| `test_modular_robot_task.py` | Module layout, atomic multi-arm submission and submodule synchronization |
| `test_workflow.py` | DAG execution, resource arbitration, failure propagation and scaling |
| `test_workflow_contracts.py` | Definitions, known input checks before dispatch, action schemas and output bindings |
| `test_workflow_cancellation.py` | Caller cancellation, node/whole-run deadlines and stop confirmation |
| `test_workflow_lifecycle.py` | Engine/adapter lifecycle, task factories and scheduler failure cleanup |
| `test_workflow_observability.py` | Wait diagnostics, event history and bounded SDK feedback |
| `test_workflow_idempotency.py` | Submission identity, replay, retention and explicit pruning |
| `test_heterogeneous_workflow.py` | Simulated arm, fixture, speech, vision and mobile-base integration |
| `test_dual_arm_examples.py` | Dual-arm example behavior and shared-zone coordination |
| `test_network_demo.py` | Network command validation, service lifecycle and demo integration |
| `test_viser_examples.py` | Optional visualization, gripper behavior and example launchers |

Run the complete suite with `python -m pytest -q`. Run related groups with:

```bash
python -m pytest -q tests/test_task.py tests/test_task_pilot.py tests/test_legacy_api.py
python -m pytest -q tests/test_schema.py tests/test_workflow_contracts.py
python -m pytest -q tests/test_robot_task.py tests/test_modular_robot_task.py
python -m pytest -q tests/test_workflow*.py tests/test_heterogeneous_workflow.py
```

Parameterize cases when the setup and assertions describe the same contract.
Keep different failure mechanisms explicit: cancellation before dispatch,
cancellation during device I/O, unknown device state and scheduler failure must
remain distinguishable. The backend fixture in `test_task.py` shares setup and
cleanup for duplicate requests, invalid IDs and bounded result retention.

Synchronize concurrent tests with events or completion callbacks and bounded
waits. Release blocked workers in `finally`, stop initialized tasks/pilots, and
close workflow engines. Use elapsed time only when timing is the behavior under
test; avoid fixed sleeps to guess when another thread has completed.

Viser tests require the optional visualization dependencies. Eager task-factory
tests require Python 3.12 or newer. Background-thread exceptions and unraisable
exceptions fail the suite, as configured in `pyproject.toml`.
