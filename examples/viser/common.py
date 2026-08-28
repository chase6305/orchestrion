"""Shared scene setup and trajectory helpers for Viser demos."""

import argparse
import concurrent.futures
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import yourdfpy

import viser
from integrations.viser.viser_modular_robot_task import (
    ViserGripperTask,
    ViserModularReducedRobotTask,
)
from orchestrion import RequestStatus, TaskPilot
from orchestrion.tasks import SubModuleTask
from viser.extras import ViserUrdf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HOME_Q = np.array(
    [np.pi / 2, -3 * np.pi / 4, 3 * np.pi / 4, -np.pi / 2, -np.pi / 2, np.pi / 2]
)
PICK_Q = np.array(
    [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2]
)
PLACE_Q = PICK_Q + np.array([np.deg2rad(35), 0.1, -0.1, 0.0, 0.0, 0.0])


def interpolate(start: np.ndarray, end: np.ndarray, steps: int) -> List[List[float]]:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    return [((1.0 - t) * start + t * end).tolist() for t in np.linspace(0, 1, steps)]


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--duration",
        type=float,
        default=-1.0,
        help="seconds to keep the scene open after the motion; default waits forever",
    )
    parser.add_argument("--startup-delay", type=float, default=0.5)
    parser.add_argument(
        "--no-wait-for-client",
        action="store_false",
        dest="wait_for_client",
        help="start immediately instead of waiting for a browser connection",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--interval", type=float, default=0.01)
    return parser


class ViserDemoRuntime:
    """Own a Viser scene, robot task, gripper task, and TaskPilot."""

    def __init__(
        self,
        title: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        scheduler_interval: float = 0.01,
    ):
        self.server = viser.ViserServer(host=host, port=port, label=title)
        self.server.gui.configure_theme(
            control_layout="fixed",
            control_width="medium",
            dark_mode=True,
            show_share_button=False,
            brand_color=(70, 110, 230),
        )
        self.server.initial_camera.position = (1.25, -1.25, 0.95)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.35)
        self.server.initial_camera.up_direction = (0.0, 0.0, 1.0)
        self.server.scene.add_grid(
            "/world/grid",
            width=2.4,
            height=2.4,
            cell_size=0.1,
            section_size=0.5,
        )
        self.server.scene.add_frame(
            "/world/base_frame", axes_length=0.2, axes_radius=0.008
        )
        self.server.gui.add_markdown("# {}\nOrchestrion v0.2 Viser demo".format(title))
        self._demo_status = self.server.gui.add_markdown(
            "**Demo:** waiting for browser client"
        )
        self._gripper_slider = self.server.gui.add_slider(
            "Gripper joint",
            min=0.0,
            max=0.725,
            step=0.005,
            initial_value=0.0,
            disabled=True,
        )
        self._gripper_status = self.server.gui.add_markdown(
            "## Peripheral health\n"
            "**gripper** · starting · idle\n\n"
            "Position `0.000` · Move `—` · Pending `0` · Failed `0`"
        )
        self._payload_status = self.server.gui.add_markdown("**Payload:** not configured")
        self._monitor_stop = threading.Event()
        self._monitor_thread = None

        arm_urdf_path = REPOSITORY_ROOT / "assets" / "UR5" / "UR5.urdf"
        arm_urdf = yourdfpy.URDF.load(str(arm_urdf_path))
        # Keep marker FK isolated from the URDF instance owned by the renderer.
        self._fk_urdf = yourdfpy.URDF.load(str(arm_urdf_path))
        gripper_urdf = yourdfpy.URDF.load(
            str(
                REPOSITORY_ROOT
                / "assets"
                / "Robotiq2F85"
                / "Robotiq2F85.urdf"
            )
        )
        arm_handle = ViserUrdf(target=self.server, urdf_or_path=arm_urdf)
        arm_handle.update_cfg(HOME_Q)
        ee_node = arm_handle._joint_frames[-1].name
        self._ee_node = ee_node
        gripper_handle = ViserUrdf(
            target=self.server,
            urdf_or_path=gripper_urdf,
            root_node_name=ee_node,
        )
        gripper_handle.update_cfg(np.zeros(1))

        main = SubModuleTask(name="arm", dof_begin=0, n_dof=6)
        gripper = SubModuleTask(name="gripper", dof_begin=6, n_dof=1)
        self.robot = ViserModularReducedRobotTask(
            init_full_q=HOME_Q.tolist() + [0.0],
            viser_handle_dict={
                "main": arm_handle,
                "sub": gripper_handle,
                "gripper": gripper_handle,
            },
            main_task=main,
            submodule_tasks=[gripper],
            interval=scheduler_interval,
        )
        self.gripper = ViserGripperTask(self.robot)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="viser-gripper"
        )
        self.pilot = TaskPilot(
            self.robot,
            {"gripper": self.gripper},
            executor_owned=self.executor,
        )
        self._pick_object = None
        self._carried_object = None
        self._placed_object = None
        self._payload_attached = False

    def set_status(self, message: str) -> None:
        self._demo_status.content = "**Demo:** {}".format(message)

    def wait_for_client(self) -> None:
        if self.server.get_clients():
            return
        connected = threading.Event()

        @self.server.on_client_connect
        def _on_connect(_client) -> None:
            connected.set()

        self.set_status("waiting for browser client")
        print(
            "Waiting for a browser client before starting the workflow...",
            flush=True,
        )
        while not connected.wait(0.1):
            if self._monitor_stop.is_set():
                return
        self.set_status("client connected; preparing workflow")

    def __enter__(self) -> "ViserDemoRuntime":
        self.pilot.initialize()
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_gripper,
            name="viser-gripper-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)
        try:
            self.pilot.stop()
        finally:
            self.server.stop()

    def _monitor_gripper(self) -> None:
        while not self._monitor_stop.wait(0.05):
            try:
                status = self.pilot.query_task_status("gripper")
            except Exception as exc:
                self._gripper_status.content = (
                    "## Peripheral health\n"
                    "**gripper** · **offline**\n\nDetail: `{}`".format(exc)
                )
                continue
            if status is None:
                self._gripper_status.content = (
                    "## Peripheral health\n"
                    "**gripper** · unknown\n\nDetail: status unsupported"
                )
                continue
            self._gripper_slider.value = status["position"]
            self._gripper_status.content = (
                "## Peripheral health\n"
                "**gripper** · {} · {}\n\n"
                "Position `{:.3f}` · Move `{}/{}` · Pending `{}` · Failed `{}`".format(
                    "online" if status["initialized"] else "offline",
                    status["activity"],
                    status["position"],
                    status["latest_finished_move_id"],
                    status["latest_move_id"],
                    status["pending"],
                    status["failed"],
                )
            )

    def add_workspace_markers(self, include_payload: bool = True) -> None:
        pick_position = self.fk_position(PICK_Q)
        place_position = self.fk_position(PLACE_Q)
        self.server.scene.add_box(
            "/world/pick",
            color=(50, 180, 80),
            dimensions=(0.16, 0.16, 0.08),
            position=pick_position,
            opacity=0.25,
        )
        self.server.scene.add_label(
            "/world/pick/label",
            "PICK",
            position=pick_position + np.array([0.0, 0.0, 0.15]),
        )
        self.server.scene.add_box(
            "/world/place",
            color=(70, 110, 230),
            dimensions=(0.16, 0.16, 0.08),
            position=place_position,
            opacity=0.25,
        )
        self.server.scene.add_label(
            "/world/place/label",
            "PLACE",
            position=place_position + np.array([0.0, 0.0, 0.15]),
        )
        if not include_payload:
            self._payload_status.content = "**Payload:** not used in this demo"
            return
        self._pick_object = self.server.scene.add_box(
            "/world/payload_at_pick",
            color=(245, 170, 40),
            dimensions=(0.07, 0.07, 0.12),
            position=pick_position,
        )
        self._carried_object = self.server.scene.add_box(
            self._ee_node + "/payload",
            color=(245, 170, 40),
            dimensions=(0.07, 0.07, 0.12),
            position=(0.0, 0.0, 0.1),
            visible=False,
        )
        self._placed_object = self.server.scene.add_box(
            "/world/payload_at_place",
            color=(245, 170, 40),
            dimensions=(0.07, 0.07, 0.12),
            position=place_position,
            visible=False,
        )
        self._payload_status.content = "**Payload:** at pick"

    def fk_position(
        self,
        joints: np.ndarray,
        local_offset: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if local_offset is None:
            local_offset = np.array([0.0, 0.0, 0.1])
        self._fk_urdf.update_cfg(joints)
        transform = self._fk_urdf.get_transform("ee_link", "base_link")
        return transform[:3, :3] @ local_offset + transform[:3, 3]

    def attach_payload(self) -> None:
        if self._pick_object is None or self._carried_object is None:
            raise RuntimeError("Call add_workspace_markers() before attaching payload")
        current = np.asarray(self.robot.query_state().jps[:6])
        if np.linalg.norm(current - PICK_Q) > 0.05:
            raise RuntimeError("Robot must be at the pick pose before attaching payload")
        gripper_position = self.robot.query_submodule_state("gripper").positions[0]
        if gripper_position < 0.65:
            raise RuntimeError("Gripper must be closed before attaching payload")
        self._pick_object.visible = False
        self._carried_object.visible = True
        self._payload_attached = True
        self._payload_status.content = "**Payload:** attached to gripper"

    def release_payload(self) -> None:
        if self._carried_object is None or self._placed_object is None:
            raise RuntimeError("Call add_workspace_markers() before releasing payload")
        if not self._payload_attached:
            return
        current = np.asarray(self.robot.query_state().jps[:6])
        if np.linalg.norm(current - PLACE_Q) > 0.05:
            raise RuntimeError("Robot must be at the place pose before releasing payload")
        gripper_position = self.robot.query_submodule_state("gripper").positions[0]
        if gripper_position > 0.08:
            raise RuntimeError("Gripper must be open before releasing payload")
        self._carried_object.visible = False
        self._placed_object.visible = True
        self._payload_attached = False
        self._payload_status.content = "**Payload:** released at place"

    def move(self, target: np.ndarray, steps: int, interval: float) -> tuple[int, int]:
        current = np.asarray(self.robot.query_state().jps[:6])
        move_ids = self.pilot.move_joint_trajectory_async(
            interpolate(current, target, steps), interval=interval
        )
        if move_ids[0] < 0:
            raise RuntimeError("Robot trajectory was rejected")
        return move_ids

    def wait_success(self, request_id: int, timeout: float = 10.0):
        result = self.pilot.wait_request(request_id, timeout=timeout)
        if result.status is not RequestStatus.SUCCEEDED:
            raise RuntimeError(
                "Request {} ended as {}: {}".format(
                    request_id, result.status.value, result.error
                )
            )
        return result

    def wait_motion(self, timeout: float = 30.0) -> None:
        if not self.pilot.wait_move(time_out=timeout):
            raise RuntimeError("Robot motion did not finish within {:.1f}s".format(timeout))

    def move_and_wait(self, target: np.ndarray, steps: int, interval: float) -> None:
        self.move(target, steps, interval)
        self.wait_motion()

    def hold(self, duration: float) -> None:
        if duration < 0:
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                return
        time.sleep(duration)


def run_demo(
    title: str,
    workflow: Callable[[ViserDemoRuntime, argparse.Namespace], None],
    argv: Optional[List[str]] = None,
) -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=title))
    args = parser.parse_args(argv)
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if not np.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive and finite")
    if not np.isfinite(args.startup_delay) or args.startup_delay < 0:
        parser.error("--startup-delay must be non-negative and finite")
    if not np.isfinite(args.duration):
        parser.error("--duration must be finite")
    with ViserDemoRuntime(
        title,
        host=args.host,
        port=args.port,
        scheduler_interval=args.interval,
    ) as runtime:
        print(
            "Open http://localhost:{} in your browser.".format(args.port),
            flush=True,
        )
        if args.host == "0.0.0.0":
            print(
                "Remote session? Use: ssh -L {0}:localhost:{0} user@host".format(
                    args.port
                ),
                flush=True,
            )
        # duration=0 is the documented headless/CI mode and must never wait for
        # a browser that will not connect.
        if args.wait_for_client and args.duration != 0:
            runtime.wait_for_client()
        time.sleep(args.startup_delay)
        runtime.set_status("workflow running")
        try:
            workflow(runtime, args)
        except Exception as exc:
            runtime.set_status("failed: {}".format(exc))
            raise
        runtime.set_status("workflow complete; server remains available")
        runtime.hold(args.duration)
