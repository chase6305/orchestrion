"""Control robot poses and gripper actions from Viser GUI buttons."""

import threading

from examples.viser.common import HOME_Q, PICK_Q, PLACE_Q, ViserDemoRuntime, run_demo
from orchestrion import MoveSyncOption


def workflow(runtime: ViserDemoRuntime, args) -> None:
    runtime.add_workspace_markers()
    status = runtime.server.gui.add_markdown("## Status\nReady")
    operation_lock = threading.Lock()

    def launch(name, operation):
        def run():
            if not operation_lock.acquire(blocking=False):
                status.content = "## Status\nBusy; ignored **{}**".format(name)
                return
            try:
                status.content = "## Status\nRunning **{}**".format(name)
                operation()
                status.content = "## Status\nCompleted **{}**".format(name)
            except Exception as exc:
                status.content = "## Status\nError: `{}`".format(exc)
            finally:
                operation_lock.release()

        threading.Thread(target=run, daemon=True).start()

    for label, target in (("Home", HOME_Q), ("Pick pose", PICK_Q), ("Place pose", PLACE_Q)):
        button = runtime.server.gui.add_button(label)
        button.on_click(
            lambda _, label=label, target=target: launch(
                label,
                lambda: runtime.move_and_wait(target, args.steps, args.interval),
            )
        )

    for action in ("open", "close"):
        button = runtime.server.gui.add_button("Gripper {}".format(action))

        def operate_gripper(action=action):
            runtime.wait_success(
                runtime.pilot.call_srv_async(
                    "gripper", {"action": action}, MoveSyncOption.no_sync()
                ),
            )
            if action == "close":
                runtime.attach_payload()
            else:
                runtime.release_payload()

        button.on_click(
            lambda _, action=action: launch(
                action,
                lambda: operate_gripper(action),
            )
        )


if __name__ == "__main__":
    run_demo("06 · Interactive Controls", workflow)
