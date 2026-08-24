# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import concurrent.futures
import logging
import math
import threading
from typing import Dict, List, Optional

import numpy as np
from viser.extras import ViserUrdf

from orchestrion.tasks.function_call_task import ThreadedPoolFunctionCallTask
from orchestrion.tasks.modular_reduced_robot_task import (
    ModularReducedRobotTask,
    SubModuleTask,
)

logger = logging.getLogger(__name__)


class ViserModularReducedRobotTask(ModularReducedRobotTask):
    """Render modular robot state updates through Viser URDF handles."""

    def __init__(
        self,
        init_full_q: List[float],
        viser_handle_dict: Dict[str, ViserUrdf],
        main_task: SubModuleTask,
        submodule_tasks: List[SubModuleTask],
        interval: float = 0.01,
    ):
        super().__init__(
            init_full_q=init_full_q,
            main_task=main_task,
            submodule_tasks=submodule_tasks,
            interval=interval,
        )
        # Store the Viser handle dictionary
        self._viser_urdf_dict = viser_handle_dict

    def _on_robot_state_bg_thread(self, jps: List[float], jps_move_id: int) -> None:
        """
        Update the Viser URDF with the current joint positions.

        Args:
            jps (List[float]): Current joint positions.
            jps_move_id (int): Move ID associated with the joint positions.
        """
        main_begin = self._main_task.dof_begin
        main_end = main_begin + self._main_task.n_dof
        self._viser_urdf_dict["main"].update_cfg(np.asarray(jps[main_begin:main_end]))
        for name, module in self._sub_task_map.items():
            handle = self._viser_urdf_dict.get(name) or self._viser_urdf_dict.get("sub")
            if handle is not None:
                begin = module.dof_begin
                handle.update_cfg(np.asarray(jps[begin : begin + module.n_dof]))

    def _on_initial_state_bg_thread(self, jps: List[float]) -> None:
        """
        Update the Viser URDF with the initial joint positions.

        Args:
            jps (List[float]): Initial joint positions.
        """
        self._on_robot_state_bg_thread(jps, -1)


class ViserGripperTask(ThreadedPoolFunctionCallTask):
    """Animate a Robotiq gripper and report completion at the target position."""

    OPEN_POSITION = 0.0
    CLOSED_POSITION = 0.72
    POSITION_LIMIT = 0.725

    def __init__(self, robot_task: ModularReducedRobotTask):
        super().__init__()
        self._robot_task = robot_task
        self._request_move_ids: Dict[int, int] = {}
        self._cancelled_requests = set()
        self._submission_lock = threading.Lock()
        self._order_condition = threading.Condition()
        self._request_sequences: Dict[int, int] = {}
        self._next_assigned_sequence = 0
        self._next_running_sequence = 0
        self._finished_sequences = set()

    def invoke_async(self, request_id: int, content: Optional[Dict] = None) -> bool:
        # Reserve sequence numbers and submit atomically. Otherwise two concurrent
        # calls (or a duplicate ID) can overwrite a queued request's sequence.
        with self._submission_lock:
            with self._lock:
                if request_id <= self._latest_sent_request:
                    return False
            with self._order_condition:
                self._request_sequences[request_id] = self._next_assigned_sequence
                self._next_assigned_sequence += 1
            try:
                accepted = super().invoke_async(request_id, content)
                if accepted:
                    return True
                with self._order_condition:
                    sequence = self._request_sequences.pop(request_id)
                    self._finish_sequence_locked(sequence)
                return False
            except Exception:
                with self._order_condition:
                    sequence = self._request_sequences.pop(request_id)
                    self._finish_sequence_locked(sequence)
                raise

    def _call_fn(self, request_id: int, content: Optional[Dict] = None) -> Dict:
        with self._order_condition:
            sequence = self._request_sequences[request_id]
            self._order_condition.wait_for(
                lambda: sequence == self._next_running_sequence
                or request_id in self._cancelled_requests
            )
        try:
            with self._lock:
                if request_id in self._cancelled_requests:
                    raise concurrent.futures.CancelledError("Gripper request cancelled")
            return self._execute_call(request_id, content)
        finally:
            with self._order_condition:
                self._finish_sequence_locked(sequence)
                self._request_sequences.pop(request_id, None)
            with self._lock:
                self._request_move_ids.pop(request_id, None)
                self._cancelled_requests.discard(request_id)

    def _finish_sequence_locked(self, sequence: int) -> None:
        self._finished_sequences.add(sequence)
        while self._next_running_sequence in self._finished_sequences:
            self._finished_sequences.remove(self._next_running_sequence)
            self._next_running_sequence += 1
        self._order_condition.notify_all()

    def _execute_call(self, request_id: int, content: Optional[Dict]) -> Dict:
        if not content:
            raise ValueError("Gripper request requires content")
        action = content.get("action")
        if "position" in content:
            target = float(content["position"])
        elif action == "open":
            target = self.OPEN_POSITION
        elif action == "close":
            target = self.CLOSED_POSITION
        else:
            raise ValueError("Use action=open/close or provide position")
        if not math.isfinite(target) or not (
            self.OPEN_POSITION <= target <= self.POSITION_LIMIT
        ):
            raise ValueError(
                "Gripper position must be in [{}, {}]".format(
                    self.OPEN_POSITION, self.POSITION_LIMIT
                )
            )
        speed = float(content.get("speed", 1.0))
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Gripper speed must be positive")

        current = self._robot_task.query_submodule_state("gripper").positions[0]
        timeout = float(content.get("timeout", abs(target - current) / speed + 1.0))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Gripper timeout must be positive")
        interval = self._robot_task.scheduler_interval
        steps = max(2, math.ceil(abs(target - current) / (speed * interval)) + 1)
        trajectory = [[float(value)] for value in np.linspace(current, target, steps)]
        move_id = self._robot_task.move_submodule_trajectory_async(
            "gripper", trajectory, interval=interval
        )
        if move_id < 0:
            raise RuntimeError("Gripper trajectory was rejected")
        with self._lock:
            self._request_move_ids[request_id] = move_id
        if not self._robot_task.wait_submodule_move("gripper", move_id, timeout):
            self._robot_task.cancel_submodule_move("gripper", move_id)
            raise concurrent.futures.CancelledError(
                "Gripper movement was cancelled or timed out"
            )
        return {
            "request_id": request_id,
            "action": action or "position",
            "position": target,
            "move_id": move_id,
        }

    def stop(self) -> None:
        with self._lock:
            pending_ids = [
                request_id
                for request_id, future in self._future_map.items()
                if not future.done()
            ]
        for request_id in pending_ids:
            self.cancel_request(request_id)
        super().stop()

    def cancel_request(self, request_id: int) -> bool:
        with self._lock:
            future = self._future_map.get(request_id)
            if future is None or future.done():
                return False
            self._cancelled_requests.add(request_id)
            move_id = self._request_move_ids.get(request_id)
        with self._order_condition:
            self._order_condition.notify_all()
        if move_id is not None:
            self._robot_task.cancel_submodule_move("gripper", move_id)
        future_cancelled = super().cancel_request(request_id)
        if future_cancelled:
            with self._order_condition:
                sequence = self._request_sequences.pop(request_id)
                self._finish_sequence_locked(sequence)
            with self._lock:
                self._request_move_ids.pop(request_id, None)
                self._cancelled_requests.discard(request_id)
        return True
