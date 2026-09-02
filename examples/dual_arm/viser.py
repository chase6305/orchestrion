"""Render any hardware-free dual-arm workflow as a two-UR5 Viser scene."""

import argparse
import bisect
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from examples.dual_arm.common import (
    LEFT_HANDOFF,
    LEFT_HOME,
    LEFT_PICK,
    RIGHT_HANDOFF,
    RIGHT_HOME,
    RIGHT_PICK,
    DualArmRuntime,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ABSTRACT_POSES = {
    "left": (LEFT_HOME, LEFT_PICK, LEFT_HANDOFF),
    "right": (RIGHT_HOME, RIGHT_PICK, RIGHT_HANDOFF),
}
_VISUAL_POSES = {
    "left": (
        [1.76258, -2.11173, 1.63838, -1.09746, -1.57082, -2.94973],
        [2.65107, -1.26360, 1.77497, -2.08215, -1.57080, -2.06127],
        [2.13826, -0.97108, 0.81157, -1.41129, -1.57080, -2.57406],
    ),
    "right": (
        [-2.66607, -2.11174, 1.63838, -1.09746, -1.57081, -1.09525],
        [-3.03257, -1.26360, 1.77498, -2.08215, -1.57080, -1.46173],
        [-2.43843, -0.97109, 0.81158, -1.41130, -1.57079, 2.27390],
    ),
}
_ROOT_POSITIONS = {
    "left": np.array([0.0, 0.55, 0.0]),
    "right": np.array([0.0, -0.55, 0.0]),
}
_PAYLOAD_LOCAL_OFFSET = np.array([0.0, 0.0, 0.08])


@dataclass(frozen=True)
class ReplayFrame:
    """One immutable visual snapshot captured from the live workflow."""

    elapsed: float
    left_arm: tuple
    right_arm: tuple
    left_gripper: float
    right_gripper: float
    zone_owner: Optional[str]
    payload_owner: str
    event: str

    @classmethod
    def from_dict(cls, content: Dict) -> "ReplayFrame":
        """Validate and restore a frame from a recording file."""
        try:
            frame = cls(
                elapsed=float(content["elapsed"]),
                left_arm=tuple(float(value) for value in content["left_arm"]),
                right_arm=tuple(float(value) for value in content["right_arm"]),
                left_gripper=float(content["left_gripper"]),
                right_gripper=float(content["right_gripper"]),
                zone_owner=content["zone_owner"],
                payload_owner=str(content["payload_owner"]),
                event=str(content["event"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid replay frame: {}".format(exc)) from exc
        if (
            frame.elapsed < 0
            or len(frame.left_arm) != 3
            or len(frame.right_arm) != 3
            or not np.all(
                np.isfinite(
                    [
                        frame.elapsed,
                        *frame.left_arm,
                        *frame.right_arm,
                        frame.left_gripper,
                        frame.right_gripper,
                    ]
                )
            )
            or frame.payload_owner
            not in {
                "table",
                "left_arm",
                "both_arms",
                "right_arm",
                "output_table",
            }
            or frame.zone_owner is not None
            and not isinstance(frame.zone_owner, str)
        ):
            raise ValueError("replay frame contains invalid state")
        return frame


def replay_frame_index(frames: List[ReplayFrame], elapsed: float) -> int:
    """Return the last frame at or before a replay timestamp."""
    if not frames:
        raise ValueError("replay requires at least one frame")
    timestamps = [frame.elapsed for frame in frames]
    return max(0, min(len(frames) - 1, bisect.bisect_right(timestamps, elapsed) - 1))


def visual_arm_configuration(
    positions: List[float], side: str = "left"
) -> np.ndarray:
    """Map an abstract arm state onto the closest smooth UR5 pose segment."""
    configuration = np.asarray(positions, dtype=float)
    if configuration.shape != (3,) or not np.all(np.isfinite(configuration)):
        raise ValueError("arm positions must contain three finite values")
    if side not in _ABSTRACT_POSES:
        raise ValueError("side must be left or right")
    abstract = tuple(np.asarray(pose, dtype=float) for pose in _ABSTRACT_POSES[side])
    visual = tuple(np.asarray(pose, dtype=float) for pose in _VISUAL_POSES[side])
    best_distance = float("inf")
    best_pose = visual[0]
    for begin_index, end_index in ((0, 1), (1, 2), (0, 2)):
        begin = abstract[begin_index]
        delta = abstract[end_index] - begin
        progress = float(
            np.clip(np.dot(configuration - begin, delta) / np.dot(delta, delta), 0, 1)
        )
        projected = begin + progress * delta
        distance = float(np.linalg.norm(configuration - projected))
        if distance < best_distance:
            best_distance = distance
            best_pose = (
                visual[begin_index]
                + progress * (visual[end_index] - visual[begin_index])
            )
    return best_pose


def world_ee_position(
    urdf: yourdfpy.URDF,
    configuration: np.ndarray,
    side: str,
    local_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute an arm end-effector position in the shared Viser world frame."""
    if side not in _ROOT_POSITIONS:
        raise ValueError("side must be left or right")
    urdf.update_cfg(np.asarray(configuration, dtype=float))
    transform = urdf.get_transform("ee_link", "base_link")
    offset = np.zeros(3) if local_offset is None else np.asarray(local_offset)
    return (
        _ROOT_POSITIONS[side]
        + transform[:3, :3] @ offset
        + transform[:3, 3]
    )


class DualArmViserRuntime(DualArmRuntime):
    """Dual-arm runtime with a live two-robot Viser representation."""

    def __init__(
        self,
        title: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        **runtime_options,
    ) -> None:
        runtime_options.setdefault("motion_time_scale", 12.0)
        super().__init__(**runtime_options)
        self.server = viser.ViserServer(host=host, port=port, label=title)
        self.server.gui.configure_theme(
            control_layout="fixed",
            control_width="medium",
            dark_mode=True,
            show_share_button=False,
            brand_color=(70, 110, 230),
        )
        self.server.initial_camera.position = (1.8, -1.8, 1.35)
        self.server.initial_camera.look_at = (0.25, 0.0, 0.35)
        self.server.initial_camera.up_direction = (0.0, 0.0, 1.0)
        self.server.scene.add_grid(
            "/world/grid", width=3.0, height=2.4, cell_size=0.1, section_size=0.5
        )
        urdf_path = REPOSITORY_ROOT / "assets" / "UR5" / "UR5.urdf"
        self._fk_urdfs = {
            side: yourdfpy.URDF.load(str(urdf_path)) for side in ("left", "right")
        }
        pick_position = self._ee_position(
            "left",
            np.asarray(_VISUAL_POSES["left"][1]),
            _PAYLOAD_LOCAL_OFFSET,
        )
        output_position = self._ee_position(
            "right",
            np.asarray(_VISUAL_POSES["right"][1]),
            _PAYLOAD_LOCAL_OFFSET,
        )
        handoff_position = 0.5 * (
            self._ee_position(
                "left",
                np.asarray(_VISUAL_POSES["left"][2]),
                _PAYLOAD_LOCAL_OFFSET,
            )
            + self._ee_position(
                "right",
                np.asarray(_VISUAL_POSES["right"][2]),
                _PAYLOAD_LOCAL_OFFSET,
            )
        )
        for side, y_position in (("left", 0.55), ("right", -0.55)):
            self.server.scene.add_cylinder(
                "/world/{}_pedestal".format(side),
                radius=0.24,
                height=0.18,
                color=(65, 72, 88),
                position=(0.0, y_position, 0.09),
            )
        for name, position in (
            ("input", pick_position),
            ("output", output_position),
        ):
            self.server.scene.add_box(
                "/world/{}_table".format(name),
                color=(75, 85, 105),
                dimensions=(0.38, 0.32, 0.16),
                position=(position[0], position[1], position[2] - 0.13),
            )
            self.server.scene.add_label(
                "/world/{}_table/label".format(name),
                name.upper(),
                position=position + np.array([0.0, 0.0, 0.16]),
            )
        self.server.scene.add_frame(
            "/world/left", position=(0.0, 0.55, 0.0), axes_length=0.15
        )
        self.server.scene.add_frame(
            "/world/right",
            position=(0.0, -0.55, 0.0),
            axes_length=0.15,
        )
        self._arm_handles = {
            side: ViserUrdf(
                target=self.server,
                urdf_or_path=yourdfpy.URDF.load(str(urdf_path)),
                root_node_name="/world/{}".format(side),
            )
            for side in ("left", "right")
        }
        self._gripper_indicators = {}
        self._gripper_fingers = {}
        self._payloads = {}
        for side, handle in self._arm_handles.items():
            ee_node = handle._joint_frames[-1].name
            self._gripper_indicators[side] = self.server.scene.add_box(
                "{}/gripper_state".format(ee_node),
                color=(70, 190, 110),
                dimensions=(0.08, 0.08, 0.025),
                position=(0.0, 0.0, 0.025),
            )
            self._gripper_fingers[side] = tuple(
                self.server.scene.add_box(
                    "{}/finger_{}".format(ee_node, index),
                    color=(70, 190, 110),
                    dimensions=(0.025, 0.025, 0.10),
                    position=(0.0, direction * 0.055, 0.075),
                )
                for index, direction in enumerate((-1.0, 1.0))
            )
            self._payloads[side] = self.server.scene.add_box(
                "{}/payload".format(ee_node),
                color=(245, 185, 55),
                dimensions=(0.09, 0.09, 0.09),
                position=_PAYLOAD_LOCAL_OFFSET,
                visible=False,
            )
        self._payloads["table"] = self.server.scene.add_box(
            "/world/payload_table",
            color=(245, 185, 55),
            dimensions=(0.09, 0.09, 0.09),
            position=pick_position,
        )
        self._payloads["both"] = self.server.scene.add_box(
            "/world/payload_handoff",
            color=(255, 205, 70),
            dimensions=(0.1, 0.1, 0.1),
            position=handoff_position,
            visible=False,
        )
        self._payloads["output"] = self.server.scene.add_box(
            "/world/payload_output",
            color=(245, 185, 55),
            dimensions=(0.09, 0.09, 0.09),
            position=output_position,
            visible=False,
        )
        self._zone_indicator = self.server.scene.add_box(
            "/world/shared_zone",
            color=(70, 190, 110),
            dimensions=(0.45, 0.45, 0.02),
            position=(handoff_position[0], handoff_position[1], 0.01),
            opacity=0.35,
        )
        self.server.scene.add_label(
            "/world/shared_zone/label",
            "SHARED ZONE",
            position=handoff_position + np.array([0.0, 0.0, 0.18]),
        )
        self.server.gui.add_markdown("# {}\nOrchestrion dual-arm workflow".format(title))
        self._workflow_status = self.server.gui.add_markdown(
            "**Workflow:** waiting for browser client"
        )
        self._state_status = self.server.gui.add_markdown("**Robot state:** starting")
        self._event_status = self.server.gui.add_markdown("## Recent events\n—")
        self._visual_stop = threading.Event()
        self._visual_thread = None
        self._capture_started_at = None
        self._replay_frames: List[ReplayFrame] = []
        self._replay_lock = threading.Lock()
        self._replay_playing = threading.Event()
        self._replay_stop = threading.Event()
        self._replay_thread = None
        self._replay_cursor = 0.0

    def _ee_position(
        self,
        side: str,
        configuration: np.ndarray,
        local_offset: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return world_ee_position(
            self._fk_urdfs[side], configuration, side, local_offset
        )

    def set_status(self, message: str) -> None:
        self._workflow_status.content = "**Workflow:** {}".format(message)

    def record(self, event: str, **details) -> None:
        super().record(event, **details)
        if not hasattr(self, "_workflow_status"):
            return
        labels = {
            "runtime_started": "initializing both controllers",
            "arms_started": "moving both arms",
            "arms_finished": "arms reached target",
            "gripper_requested": "operating {} gripper".format(
                details.get("side", "")
            ),
            "payload_owner_changed": "payload: {}".format(
                details.get("owner", "unknown").replace("_", " ")
            ),
            "zone_requested": "shared zone: {}".format(
                details.get("action", "updating")
            ),
            "arms_aborted": "coordinated stop complete",
            "inspection_requested": "remote inspection running",
            "inspection_finished": "remote inspection complete",
        }
        if event in labels:
            self.set_status(labels[event])

    def wait_for_client(self) -> None:
        if self.server.get_clients():
            return
        connected = threading.Event()

        @self.server.on_client_connect
        def _on_connect(_client) -> None:
            connected.set()

        self.set_status("waiting for browser client")
        print("Waiting for a browser client before starting the workflow...", flush=True)
        while not connected.wait(0.1):
            if self._visual_stop.is_set():
                return
        self.set_status("client connected; running")

    def __enter__(self) -> "DualArmViserRuntime":
        try:
            super().__enter__()
        except Exception:
            self.server.stop()
            raise
        self._visual_stop.clear()
        self._visual_thread = threading.Thread(
            target=self._monitor_scene,
            name="dual-arm-viser-monitor",
            daemon=True,
        )
        self._visual_thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            arms = self.arm_positions()
            grippers = {
                side: self.robot.query_submodule_state(
                    "{}_gripper".format(side)
                ).positions[0]
                for side in ("left", "right")
            }
            frame = self._capture_frame(arms, grippers, self.zone.status()["owner"])
            self._render_frame(frame)
        except Exception:
            pass
        self._visual_stop.set()
        if self._visual_thread is not None:
            self._visual_thread.join(timeout=1.0)
        super().__exit__(exc_type, exc, traceback)

    def close_server(self) -> None:
        """Stop the Viser transport after the completed scene has been viewed."""
        self._replay_stop.set()
        self._replay_playing.set()
        if self._replay_thread is not None:
            self._replay_thread.join(timeout=1.0)
        self.server.stop()

    @property
    def replay_frames(self) -> List[ReplayFrame]:
        with self._replay_lock:
            return list(self._replay_frames)

    def save_replay(self, path: Path, title: str) -> None:
        """Persist captured frames as a portable, versioned JSON recording."""
        frames = self.replay_frames
        if not frames:
            raise ValueError("cannot save an empty replay")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "format": "orchestrion-dual-arm-replay",
                    "version": 1,
                    "title": title,
                    "duration": frames[-1].elapsed,
                    "frames": [asdict(frame) for frame in frames],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_replay(self, path: Path) -> Dict:
        """Load and validate a recording without initializing robot services."""
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot load replay {}: {}".format(path, exc)) from exc
        if (
            not isinstance(content, dict)
            or content.get("format") != "orchestrion-dual-arm-replay"
            or content.get("version") != 1
            or not isinstance(content.get("frames"), list)
            or not content["frames"]
        ):
            raise ValueError("unsupported or empty replay recording")
        frames = [ReplayFrame.from_dict(frame) for frame in content["frames"]]
        if any(
            current.elapsed < previous.elapsed
            for previous, current in zip(frames, frames[1:])
        ):
            raise ValueError("replay timestamps must be monotonic")
        with self._replay_lock:
            self._replay_frames = frames
            self._replay_cursor = frames[-1].elapsed
        self._render_frame(frames[-1], replaying=True)
        return {
            "title": str(content.get("title", path.stem)),
            "frames": len(frames),
            "duration": frames[-1].elapsed,
        }

    def _payload_owner(self) -> str:
        owners = [
            event["owner"]
            for event in self.events
            if event["event"] == "payload_owner_changed"
        ]
        return owners[-1] if owners else "table"

    def _capture_frame(
        self,
        arms: Dict[str, List[float]],
        grippers: Dict[str, float],
        zone_owner: Optional[str],
    ) -> ReplayFrame:
        now = time.monotonic()
        if self._capture_started_at is None:
            self._capture_started_at = now
        events = self.events
        frame = ReplayFrame(
            elapsed=now - self._capture_started_at,
            left_arm=tuple(arms["left"]),
            right_arm=tuple(arms["right"]),
            left_gripper=grippers["left"],
            right_gripper=grippers["right"],
            zone_owner=zone_owner,
            payload_owner=self._payload_owner(),
            event=events[-1]["event"] if events else "starting",
        )
        with self._replay_lock:
            self._replay_frames.append(frame)
        return frame

    def _render_frame(self, frame: ReplayFrame, replaying: bool = False) -> None:
        ee_positions = {}
        for side, positions in (
            ("left", frame.left_arm),
            ("right", frame.right_arm),
        ):
            configuration = visual_arm_configuration(list(positions), side)
            self._arm_handles[side].update_cfg(configuration)
            ee_positions[side] = self._ee_position(
                side, configuration, _PAYLOAD_LOCAL_OFFSET
            )
            grip = getattr(frame, "{}_gripper".format(side))
            self._gripper_indicators[side].color = (
                (230, 145, 55) if grip >= 0.5 else (70, 190, 110)
            )
            finger_offset = 0.025 + 0.03 * (1.0 - grip)
            for finger, direction in zip(
                self._gripper_fingers[side], (-1.0, 1.0)
            ):
                finger.position = (0.0, direction * finger_offset, 0.075)
                finger.color = self._gripper_indicators[side].color
        self._zone_indicator.color = (
            (225, 85, 70) if frame.zone_owner else (70, 190, 110)
        )
        visibility = {
            "table": frame.payload_owner == "table",
            "left": frame.payload_owner == "left_arm",
            "both": frame.payload_owner == "both_arms",
            "right": frame.payload_owner == "right_arm",
            "output": frame.payload_owner == "output_table",
        }
        for payload, visible in visibility.items():
            self._payloads[payload].visible = visible
        self._payloads["both"].position = 0.5 * (
            ee_positions["left"] + ee_positions["right"]
        )
        mode = "Replay" if replaying else "Live"
        self._state_status.content = (
            "## {} state · {:.2f}s\n"
            "Left `{}` · Right `{}`\n\n"
            "Shared zone: **{}** · Payload: **{}**\n\nStage: `{}`"
        ).format(
            mode,
            frame.elapsed,
            ", ".join("{:.2f}".format(value) for value in frame.left_arm),
            ", ".join("{:.2f}".format(value) for value in frame.right_arm),
            frame.zone_owner or "available",
            frame.payload_owner.replace("_", " "),
            frame.event,
        )

    def _monitor_scene(self) -> None:
        while not self._visual_stop.wait(0.03):
            try:
                arms = self.arm_positions()
                grippers = {
                    side: self.robot.query_submodule_state(
                        "{}_gripper".format(side)
                    ).positions[0]
                    for side in ("left", "right")
                }
                zone = self.zone.status()
                frame = self._capture_frame(arms, grippers, zone["owner"])
                self._render_frame(frame)
                recent = self.events[-6:]
                self._event_status.content = "## Recent events\n{}".format(
                    "\n".join("- `{}`".format(event["event"]) for event in recent)
                    or "—"
                )
            except Exception as exc:
                self._state_status.content = "**Robot state:** unavailable (`{}`)".format(
                    exc
                )

    def enable_replay(
        self,
        auto_play: bool = False,
        loop: bool = False,
        initial_speed: float = 1.0,
    ) -> None:
        """Add replay controls and optionally start playing the captured run."""
        frames = self.replay_frames
        if not frames:
            self.set_status("complete; no replay frames captured")
            return
        duration = frames[-1].elapsed
        self.server.gui.add_markdown("## Replay controls")
        timeline = self.server.gui.add_slider(
            "Timeline (s)",
            min=0.0,
            max=max(duration, 0.01),
            step=max(duration / max(len(frames) - 1, 1), 0.001),
            initial_value=duration,
        )
        speed = self.server.gui.add_slider(
            "Playback speed",
            min=0.25,
            max=3.0,
            step=0.25,
            initial_value=initial_speed,
        )
        loop_control = self.server.gui.add_checkbox("Loop replay", initial_value=loop)
        play = self.server.gui.add_button("Play", color="green")
        pause = self.server.gui.add_button("Pause")
        restart = self.server.gui.add_button("Restart", color="blue")

        def seek(elapsed: float) -> None:
            with self._replay_lock:
                self._replay_cursor = float(np.clip(elapsed, 0.0, duration))
                cursor = self._replay_cursor
            self._render_frame(frames[replay_frame_index(frames, cursor)], replaying=True)

        @timeline.on_update
        def _seek_from_slider(_event) -> None:
            seek(float(timeline.value))

        @play.on_click
        def _play(_event) -> None:
            if self._replay_cursor >= duration:
                seek(0.0)
            self._replay_playing.set()
            self.set_status("replaying")

        @pause.on_click
        def _pause(_event) -> None:
            self._replay_playing.clear()
            self.set_status("replay paused")

        @restart.on_click
        def _restart(_event) -> None:
            seek(0.0)
            timeline.value = 0.0
            self._replay_playing.set()
            self.set_status("replaying from start")

        def replay_worker() -> None:
            last_tick = time.monotonic()
            while not self._replay_stop.wait(0.02):
                now = time.monotonic()
                delta = now - last_tick
                last_tick = now
                if not self._replay_playing.is_set():
                    continue
                with self._replay_lock:
                    self._replay_cursor += delta * float(speed.value)
                    cursor = self._replay_cursor
                if cursor >= duration:
                    if loop_control.value:
                        seek(0.0)
                        cursor = 0.0
                    else:
                        seek(duration)
                        timeline.value = duration
                        self._replay_playing.clear()
                        self.set_status("replay complete")
                        continue
                timeline.value = cursor
                self._render_frame(
                    frames[replay_frame_index(frames, cursor)], replaying=True
                )

        self._replay_stop.clear()
        self._replay_thread = threading.Thread(
            target=replay_worker, name="dual-arm-replay", daemon=True
        )
        self._replay_thread.start()
        if auto_play:
            seek(0.0)
            timeline.value = 0.0
            self._replay_playing.set()
            self.set_status("replaying captured workflow")
        else:
            self.set_status("complete; replay ready")


def _wait_after_workflow(duration: float) -> None:
    if duration < 0:
        print("Workflow complete. Press Ctrl+C to stop the Viser server.", flush=True)
        while True:
            time.sleep(1.0)
    if duration:
        time.sleep(duration)


def run_demo(
    title: str,
    workflow: Callable[..., Dict],
    argv: Optional[List[str]] = None,
) -> Dict:
    parser = argparse.ArgumentParser(description="Render {} in Viser.".format(title))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--duration",
        type=float,
        default=-1.0,
        help="seconds to keep the scene open; default waits forever",
    )
    parser.add_argument(
        "--no-wait-for-client",
        action="store_false",
        dest="wait_for_client",
        help="start immediately instead of waiting for a browser connection",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="automatically replay the captured workflow after it completes",
    )
    parser.add_argument(
        "--loop-replay",
        action="store_true",
        help="loop playback (also enables automatic replay)",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        choices=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
        help="initial replay speed",
    )
    parser.add_argument(
        "--recording-out",
        type=Path,
        help="save the captured workflow as a portable JSON replay",
    )
    parser.add_argument(
        "--replay-file",
        type=Path,
        help="load a saved recording instead of executing the workflow",
    )
    args = parser.parse_args(argv)
    holder: Dict[str, DualArmViserRuntime] = {}

    def runtime_factory(**runtime_options) -> DualArmViserRuntime:
        runtime = DualArmViserRuntime(
            title, host=args.host, port=args.port, **runtime_options
        )
        holder["runtime"] = runtime
        if args.wait_for_client:
            runtime.wait_for_client()
        runtime.set_status("running")
        return runtime

    try:
        if args.replay_file is not None:
            runtime = DualArmViserRuntime(title, host=args.host, port=args.port)
            holder["runtime"] = runtime
            if args.wait_for_client:
                runtime.wait_for_client()
            metadata = runtime.load_replay(args.replay_file)
            runtime.enable_replay(
                auto_play=True,
                loop=args.loop_replay,
                initial_speed=args.replay_speed,
            )
            print(json.dumps({"replay": metadata}, indent=2))
            _wait_after_workflow(args.duration)
            return {"replay": metadata}
        summary = workflow(runtime_factory=runtime_factory)
        runtime = holder.get("runtime")
        if runtime is not None:
            if args.recording_out is not None:
                runtime.save_replay(args.recording_out, title)
                summary["recording"] = str(args.recording_out)
            runtime.enable_replay(
                auto_play=args.replay or args.loop_replay,
                loop=args.loop_replay,
                initial_speed=args.replay_speed,
            )
        print(json.dumps(summary, indent=2))
        _wait_after_workflow(args.duration)
        return summary
    except KeyboardInterrupt:
        return {}
    finally:
        runtime = holder.get("runtime")
        if runtime is not None:
            runtime.close_server()
