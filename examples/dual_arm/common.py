"""Reusable runtime and services for hardware-free dual-arm demos."""

import concurrent.futures
import copy
import threading
import time
from typing import Dict, List, Optional

from orchestrion import (
    CallableTask,
    FieldSpec,
    MoveSyncOption,
    RequestSchema,
    TaskPilot,
)
from orchestrion.tasks import GenericTask, ModularReducedRobotTask, SubModuleTask

LEFT_HOME = [0.0, 0.25, -0.4]
RIGHT_HOME = [0.0, -0.25, 0.4]
LEFT_PICK = [0.55, 0.15, -0.65]
RIGHT_PICK = [0.55, -0.15, 0.65]
LEFT_HANDOFF = [0.35, -0.05, -0.25]
RIGHT_HANDOFF = [0.35, 0.05, 0.25]


def interpolate(start: List[float], target: List[float], steps: int) -> List[List[float]]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if len(start) != len(target):
        raise ValueError("start and target must have the same dimensions")
    return [
        [
            begin + (end - begin) * step / steps
            for begin, end in zip(start, target)
        ]
        for step in range(1, steps + 1)
    ]


class DualArmRobot(ModularReducedRobotTask):
    """Nine-DOF robot: torso, two 3-DOF arms, and two grippers."""

    def __init__(self, interval: float = 0.002) -> None:
        super().__init__(
            [0.0, *LEFT_HOME, *RIGHT_HOME, 0.0, 0.0],
            SubModuleTask("torso", 0, 1),
            [
                SubModuleTask("left_arm", 1, 3),
                SubModuleTask("right_arm", 4, 3),
                SubModuleTask("left_gripper", 7, 1),
                SubModuleTask("right_gripper", 8, 1),
            ],
            interval=interval,
        )
        self._observed_states: List[List[float]] = []

    @property
    def observed_states(self) -> List[List[float]]:
        with self._lock:
            return [state.copy() for state in self._observed_states]

    def _on_robot_state_bg_thread(self, jps: List[float], jps_move_id: int):
        with self._lock:
            self._observed_states.append(jps.copy())

    def _on_initial_state_bg_thread(self, jps: List[float]):
        with self._lock:
            self._observed_states.append(jps.copy())


class SharedZoneCoordinator:
    """Reserve a shared workspace for one coordinated cycle at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: Optional[str] = None
        self._revision = 0

    def command(self, request_id: int, content: Dict) -> Dict:
        action = content["action"]
        cycle = content["cycle"]
        if action not in {"reserve", "release"}:
            raise ValueError("shared-zone action must be reserve or release")
        if not isinstance(cycle, str) or not cycle:
            raise ValueError("shared-zone cycle must be a non-empty string")
        with self._lock:
            if action == "reserve":
                if self._owner not in (None, cycle):
                    raise RuntimeError(
                        "shared zone is reserved by cycle {}".format(self._owner)
                    )
                changed = self._owner is None
                self._owner = cycle
            else:
                if self._owner != cycle:
                    raise RuntimeError("cycle {} does not own shared zone".format(cycle))
                changed = True
                self._owner = None
            if changed:
                self._revision += 1
            return {
                "request_id": request_id,
                "action": action,
                "cycle": cycle,
                "owner": self._owner,
                "revision": self._revision,
            }

    def status(self) -> Dict:
        with self._lock:
            return {
                "health": "online",
                "available": True,
                "observed_at": time.time(),
                "owner": self._owner,
                "revision": self._revision,
            }


class DualArmRuntime:
    """Own a dual-arm robot, gripper services, and shared-zone service."""

    def __init__(
        self,
        interval: float = 0.002,
        motion_time_scale: float = 1.0,
        extra_services: Optional[Dict[str, GenericTask]] = None,
    ) -> None:
        if motion_time_scale <= 0:
            raise ValueError("motion_time_scale must be positive")
        self.motion_time_scale = motion_time_scale
        if extra_services is not None and (
            not isinstance(extra_services, dict)
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(task, GenericTask)
                for name, task in extra_services.items()
            )
        ):
            raise TypeError("extra_services must map names to GenericTask instances")
        self.robot = DualArmRobot(interval=interval)
        self.zone = SharedZoneCoordinator()
        self._events: List[Dict] = []
        self._event_lock = threading.Lock()
        services = {
            "left_gripper": self._make_gripper_service("left_gripper"),
            "right_gripper": self._make_gripper_service("right_gripper"),
            "shared_zone": CallableTask(
                self.zone.command,
                status=self.zone.status,
                request_schema=RequestSchema(
                    {
                        "action": FieldSpec(
                            "string", required=True, choices=("reserve", "release")
                        ),
                        "cycle": FieldSpec("string", required=True),
                    }
                ),
                metadata={"resource": "shared_handoff_zone"},
            ),
        }
        if extra_services:
            conflicts = set(services).intersection(extra_services)
            if conflicts:
                raise ValueError(
                    "duplicate runtime services: {}".format(
                        ", ".join(sorted(conflicts))
                    )
                )
            services.update(extra_services)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.pilot = TaskPilot(
            self.robot, services, executor_owned=self._executor, poll_interval=0.002
        )

    @property
    def events(self) -> List[Dict]:
        with self._event_lock:
            return copy.deepcopy(self._events)

    def __enter__(self) -> "DualArmRuntime":
        try:
            self.pilot.initialize()
        except Exception:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise
        self.record("runtime_started")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.record("runtime_stopping")
        self.pilot.stop()

    def record(self, event: str, **details) -> None:
        with self._event_lock:
            self._events.append(
                copy.deepcopy({"event": event, "timestamp": time.time(), **details})
            )

    def move_arms(
        self,
        left_target: List[float],
        right_target: List[float],
        steps: int = 20,
        interval: float = 0.003,
    ) -> Dict[str, int]:
        left_start = self.robot.query_submodule_state("left_arm").positions
        right_start = self.robot.query_submodule_state("right_arm").positions
        moves = self.robot.move_submodules_trajectories_async(
            {
                "left_arm": interpolate(left_start, left_target, steps),
                "right_arm": interpolate(right_start, right_target, steps),
            },
            interval * self.motion_time_scale,
        )
        if set(moves) != {"left_arm", "right_arm"}:
            raise RuntimeError("dual-arm trajectory was rejected")
        self.record("arms_started", moves=moves.copy())
        return moves

    def wait_arms(self, moves: Dict[str, int], timeout: float = 2.0) -> None:
        if not self.robot.wait_submodule_moves(moves, timeout=timeout):
            cancelled = self.abort_arms(moves)
            raise RuntimeError(
                "dual-arm movement failed; peer cancellation: {}".format(cancelled)
            )
        self.record("arms_finished", moves=moves.copy())

    def abort_arms(self, moves: Dict[str, int]) -> Dict[str, bool]:
        cancelled = self.robot.cancel_submodule_moves(moves)
        self.record("arms_aborted", moves=moves.copy(), cancelled=cancelled.copy())
        return cancelled

    def command_gripper(
        self,
        side: str,
        action: str,
        sync_option: Optional[MoveSyncOption] = None,
    ) -> int:
        service = "{}_gripper".format(side)
        selected_sync = (
            MoveSyncOption.no_sync() if sync_option is None else sync_option
        )
        request_id = self.pilot.call_srv_async(
            service,
            {"action": action},
            selected_sync,
        )
        self.record("gripper_requested", side=side, action=action, request_id=request_id)
        return request_id

    def command_zone(self, action: str, cycle: str) -> int:
        request_id = self.pilot.call_srv_async(
            "shared_zone",
            {"action": action, "cycle": cycle},
            MoveSyncOption.no_sync(),
        )
        self.record("zone_requested", action=action, cycle=cycle, request_id=request_id)
        return request_id

    def arm_positions(self) -> Dict[str, List[float]]:
        states = self.robot.query_submodule_states(["left_arm", "right_arm"])
        return {
            side: states["{}_arm".format(side)].positions
            for side in ("left", "right")
        }

    def _make_gripper_service(self, module_name: str) -> CallableTask:
        def command(request_id: int, content: Dict) -> Dict:
            target = 1.0 if content["action"] == "close" else 0.0
            start = self.robot.query_submodule_state(module_name).positions
            move_id = self.robot.move_submodule_trajectory_async(
                module_name,
                interpolate(start, [target], 16),
                interval=0.002 * self.motion_time_scale,
            )
            if move_id < 0 or not self.robot.wait_submodule_move(
                module_name, move_id, timeout=1.0
            ):
                raise RuntimeError("{} movement failed".format(module_name))
            self.record(
                "gripper_finished",
                module=module_name,
                action=content["action"],
                request_id=request_id,
            )
            return {"action": content["action"], "position": target}

        return CallableTask(
            command,
            request_schema=RequestSchema(
                {
                    "action": FieldSpec(
                        "string", required=True, choices=("open", "close")
                    )
                }
            ),
            metadata={"module": module_name, "commands": ["open", "close"]},
        )


def final_snapshot(runtime: DualArmRuntime) -> Dict:
    """Return a JSON-compatible state shared by all demo summaries."""
    return {
        "arms": runtime.arm_positions(),
        "left_gripper": runtime.robot.query_submodule_state(
            "left_gripper"
        ).positions[0],
        "right_gripper": runtime.robot.query_submodule_state(
            "right_gripper"
        ).positions[0],
        "zone": runtime.zone.status(),
        "health": runtime.pilot.query_health(stale_after=1.0),
        "events": runtime.events,
    }
