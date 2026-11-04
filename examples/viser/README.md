# Orchestrion Viser Demos

Install the visualization dependencies from the repository root:

```bash
python -m pip install -e ".[viser]"
```

Each command starts a browser-based Viser server at `http://localhost:8080`, waits
for the first browser client, then plays the workflow. The server remains open until
Ctrl+C. Use `--port` to select another port and `--steps`/`--interval` to control
animation detail and speed. `--no-wait-for-client` starts immediately.

When running on a remote machine, forward the port first:

```bash
ssh -L 8080:localhost:8080 user@remote-host
```

Then open `http://localhost:8080` on your local computer. The preferred launcher is:

```bash
python -m examples.viser --list
python -m examples.viser 02
```

| Demo | What it shows |
| --- | --- |
| 01 Joint Sweep | Smooth joint interpolation and return home |
| 02 Pick and Place | Move-synchronized close/open actions |
| 03 Segmented Sync | Actions attached to two segment move IDs |
| 04 Parallel Motion | Arm and gripper executing concurrently |
| 05 Request Timeline | Live request states rendered in the GUI |
| 06 Interactive Controls | GUI buttons for poses and gripper actions |
| 07 Motion Trail | Animated robot plus its Cartesian end-effector path |
| 08 Gripper Lab | Interactive position, speed, reversal, and cancellation |

The repository README features a real runtime recording of demo 02. Green marks
PICK and blue marks PLACE; the yellow payload follows the gripper only after
closure completes. The panel shows the four phases and peripheral health.

For a quick headless smoke run:

```bash
python -m examples.viser 02 \
  --startup-delay 0 --duration 0 --no-wait-for-client \
  --steps 2 --interval 0.001
```

## Gripper Commands

Gripper requests complete only after the animated finger joint reaches its target.
Trajectories always start from the current position, so repeated commands and motion
reversal do not jump back to a hard-coded endpoint.

```python
pilot.call_srv_async("gripper", {"action": "open", "speed": 0.8})
pilot.call_srv_async("gripper", {"action": "close", "speed": 0.5})
pilot.call_srv_async(
    "gripper",
    {"position": 0.35, "speed": 0.6, "timeout": 2.0},
)
```

`position` is the Robotiq URDF driver-joint angle in radians and must be within
`[0.0, 0.725]`. Running requests support best-effort cancellation through
`TaskPilot.cancel_request()`.
