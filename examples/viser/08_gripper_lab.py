"""Interactive gripper position, speed, reversal, and cancellation laboratory."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import threading

from examples.viser.common import ViserDemoRuntime, run_demo
from orchestrion import MoveSyncOption


def workflow(runtime: ViserDemoRuntime, args) -> None:
    runtime.add_workspace_markers(include_payload=False)
    target = runtime.server.gui.add_slider(
        "Target joint position",
        min=0.0,
        max=0.725,
        step=0.005,
        initial_value=0.35,
    )
    speed = runtime.server.gui.add_slider(
        "Speed (rad/s)", min=0.05, max=2.0, step=0.05, initial_value=0.5
    )
    status = runtime.server.gui.add_markdown("## Command\nIdle")
    state_lock = threading.Lock()
    active_request = {"id": None}

    def command(content):
        def run():
            request_id = runtime.pilot.call_srv_async(
                "gripper", content, MoveSyncOption.no_sync()
            )
            with state_lock:
                active_request["id"] = request_id
            status.content = "## Command\nRequest `{}` running".format(request_id)
            result = runtime.pilot.wait_request(request_id)
            with state_lock:
                is_latest = active_request["id"] == request_id
            if is_latest:
                status.content = "## Command\nRequest `{}` **{}**{}".format(
                    request_id,
                    result.status.value,
                    " · " + result.error if result.error else "",
                )

        threading.Thread(target=run, daemon=True).start()

    move_button = runtime.server.gui.add_button("Move to target", color="blue")
    move_button.on_click(
        lambda _: command({"position": target.value, "speed": speed.value})
    )
    open_button = runtime.server.gui.add_button("Open")
    open_button.on_click(lambda _: command({"action": "open", "speed": speed.value}))
    close_button = runtime.server.gui.add_button("Close")
    close_button.on_click(
        lambda _: command({"action": "close", "speed": speed.value})
    )

    cancel_button = runtime.server.gui.add_button("Cancel active", color="red")

    def cancel(_):
        with state_lock:
            request_id = active_request["id"]
        if request_id is None:
            status.content = "## Command\nNo active request"
            return
        result = runtime.pilot.query_request(request_id)
        if result.status.terminal:
            status.content = "## Command\nRequest already {}".format(result.status.value)
        elif runtime.pilot.cancel_request(request_id):
            status.content = "## Command\nCancellation requested for `{}`".format(
                request_id
            )

    cancel_button.on_click(cancel)

    reverse_button = runtime.server.gui.add_button("Cancel and reverse", color="orange")

    def reverse(_):
        with state_lock:
            request_id = active_request["id"]
        if request_id is not None:
            result = runtime.pilot.query_request(request_id)
            if not result.status.terminal:
                runtime.pilot.cancel_request(request_id)
        position = runtime.robot.query_submodule_state("gripper").positions[0]
        reverse_target = 0.0 if position > 0.1 else 0.72
        command({"position": reverse_target, "speed": speed.value})

    reverse_button.on_click(reverse)


if __name__ == "__main__":
    run_demo("08 · Gripper Laboratory", workflow)
