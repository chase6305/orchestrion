# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import copy
import numpy as np
from typing import List, Optional, Dict

import viser
from viser.extras import ViserUrdf

from orchestrion.tasks.generic_task import GenericTask
from orchestrion.tasks.modular_reduced_robot_task import (
    SubModuleTask,
    ModularReducedRobotTask,
)

import logging

logger = logging.getLogger(__name__)


class ViserModularReducedRobotTask(ModularReducedRobotTask):
    """
    ViserModularReducedRobotTask is a placeholder for the actual robot task implementation.
    This class should implement methods to communicate with the Viser robot.
    """

    def __init__(
        self,
        init_full_q: List[float],
        viser_handle_dict: Dict[str, ViserUrdf],
        main_task: SubModuleTask,
        submodule_tasks: List[SubModuleTask],
    ):
        super().__init__(
            init_full_q=init_full_q,
            main_task=main_task,
            submodule_tasks=submodule_tasks,
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
        self._viser_urdf_dict["main"].update_cfg(np.array(jps[:6]))
        self._viser_urdf_dict["sub"].update_cfg(np.array(jps[6:]))

    def _on_initial_state_bg_thread(self, jps: List[float]) -> None:
        """
        Update the Viser URDF with the initial joint positions.

        Args:
            jps (List[float]): Initial joint positions.
        """
        self._viser_urdf_dict["main"].update_cfg(np.array(jps[:6]))
        self._viser_urdf_dict["sub"].update_cfg(np.array(jps[6:]))


class ViserGripperTask(GenericTask):
    """
    ViserGripperTask is a placeholder for the actual gripper task implementation.
    This class should implement methods to communicate with the gripper.
    """

    def __init__(self, robot_task: ModularReducedRobotTask):
        self._robot_task = robot_task

    def invoke_async(self, request_id: int, content: Optional[Dict] = None) -> bool:
        if content and "action" in content:
            open_steps = 50
            close_traj = [
                [(1 - t) * 0.0 + t * 1.0] for t in np.linspace(0, 0.72, open_steps)
            ]
            open_traj = copy.deepcopy(close_traj)
            open_traj = open_traj[::-1]
            action = content["action"]
            if action == "open":
                # Implement gripper open logic
                self._robot_task.move_submodule_async(
                    "gripper", open_traj, interval=0.01
                )
            elif action == "close":
                # Implement gripper close logic
                self._robot_task.move_submodule_async(
                    "gripper", close_traj, interval=0.01
                )
            else:
                return False
            return True
