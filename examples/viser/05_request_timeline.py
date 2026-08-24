"""Display a live Orchestrion request timeline in the Viser GUI."""

import time

from examples.viser.common import PICK_Q, ViserDemoRuntime, run_demo
from orchestrion import MoveSyncOption


def workflow(runtime: ViserDemoRuntime, args) -> None:
    panel = runtime.server.gui.add_markdown("## Request timeline\nWaiting for request…")
    runtime.set_status("recording synchronized request timeline")
    _, move_end = runtime.move(PICK_Q, args.steps, args.interval)
    request_id = runtime.pilot.call_srv_async(
        "gripper",
        {"action": "close"},
        MoveSyncOption.sync_w_explicit_id(move_end - 1),
    )
    while True:
        result = runtime.pilot.query_request(request_id)
        lines = ["## Request timeline"]
        events = runtime.pilot.timeline(request_id)
        started_at = events[0].timestamp
        lines.extend(
            "- `+{:.3f}s` **{}**".format(
                event.timestamp - started_at, event.status.value
            )
            for event in events
        )
        panel.content = "\n".join(lines)
        if result.status.terminal:
            break
        time.sleep(0.01)
    runtime.wait_success(request_id)


if __name__ == "__main__":
    run_demo("05 · Live Request Timeline", workflow)
