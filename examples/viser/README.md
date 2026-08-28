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

The original module commands and direct commands such as
`python examples/viser/02_pick_and_place.py` are also supported.

| Demo | What it shows | Command |
| --- | --- | --- |
| 01 Joint Sweep | Smooth joint interpolation and return home | `python -m examples.viser 01` |
| 02 Pick and Place | Move-synchronized close/open actions | `python -m examples.viser 02` |
| 03 Segmented Sync | Actions attached to two segment move IDs | `python -m examples.viser 03` |
| 04 Parallel Motion | Arm and gripper executing concurrently | `python -m examples.viser 04` |
| 05 Request Timeline | Live request states rendered in the GUI | `python -m examples.viser 05` |
| 06 Interactive Controls | GUI buttons for poses and gripper actions | `python -m examples.viser 06` |
| 07 Motion Trail | Animated robot plus its Cartesian end-effector path | `python -m examples.viser 07` |
| 08 Gripper Lab | Interactive position, speed, reversal, and cancellation | `python -m examples.viser 08` |

The repository README includes a real runtime recording of demo 02 at
`assets/02-pick-and-place.webp`.

For a quick headless smoke run:

```bash
python -m examples.viser.02_pick_and_place \
  --startup-delay 0 --duration 0 --no-wait-for-client \
  --steps 2 --interval 0.001
```

The legacy command `python -m examples.viser.viser_modular_robot_task` remains
available and runs the pick-and-place workflow.

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
