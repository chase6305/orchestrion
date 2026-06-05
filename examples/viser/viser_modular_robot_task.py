"""
Viser Modular Robot Task Example

requirements:
- viser
- yourdfpy

This example demonstrates how to use the ViserModularReducedRobotTask to control a modular robot in Viser.
It sets up a UR10 arm with a gripper, generates trajectories for picking and placing an object, and executes the trajectories while synchronizing gripper actions.
"""

import time
import concurrent.futures
from pathlib import Path
import numpy as np

import viser
import yourdfpy
from viser.extras import ViserUrdf

from orchestrion.move_sync_option import MoveSyncOption
from orchestrion.task_pilot import TaskPilot
from orchestrion.tasks.modular_reduced_robot_task import SubModuleTask
from integrations.viser.viser_modular_robot_task import (
    ViserGripperTask,
    ViserModularReducedRobotTask,
)

if __name__ == "__main__":

    viser_server = viser.ViserServer()
    arm_urdf_path = str(
        Path(__file__).resolve().parent.parent.parent / "assets" / "UR5" / "UR5.urdf"
    )
    arm_urdf = yourdfpy.URDF.load(arm_urdf_path)

    gripper_urdf_path = str(
        Path(__file__).resolve().parent.parent.parent
        / "assets"
        / "Robotiq2F85"
        / "Robotiq2F85.urdf"
    )
    gripper_urdf = yourdfpy.URDF.load(gripper_urdf_path)

    init_qpos = np.array(
        [np.pi / 2, -np.pi / 4 * 3, np.pi / 4 * 3, -np.pi / 2, -np.pi / 2, np.pi / 2]
    )
    target_qpos = np.array(
        [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2]
    )
    arm_viser_handle = ViserUrdf(target=viser_server, urdf_or_path=arm_urdf)
    arm_viser_handle.update_cfg(init_qpos)

    ee_link_node = arm_viser_handle._joint_frames[-1].name
    gripper_viser_handle = ViserUrdf(
        target=viser_server, urdf_or_path=gripper_urdf, root_node_name=ee_link_node
    )
    gripper_viser_handle.update_cfg(np.zeros(1))

    viser_handle_dict = {"main": arm_viser_handle, "sub": gripper_viser_handle}

    init_full_q = init_qpos.tolist() + [0.0]
    main_arm_module = SubModuleTask()
    main_arm_module.name = "arm"
    main_arm_module.dof_begin = 0
    main_arm_module.n_dof = 6
    sub_gripper_module = SubModuleTask()
    sub_gripper_module.name = "gripper"
    sub_gripper_module.dof_begin = 6
    sub_gripper_module.n_dof = 1

    task = ViserModularReducedRobotTask(
        init_full_q=init_full_q,
        viser_handle_dict=viser_handle_dict,
        main_task=main_arm_module,
        submodule_tasks=[sub_gripper_module],
    )
    task.initialize()

    gripper_task = ViserGripperTask(robot_task=task)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    supervisior = TaskPilot(
        robot_task=task,
        task_map={"gripper": gripper_task},
        executor_owned=executor,
    )
    supervisior.initialize()

    # Generate trajectory from init_qpos to target_qpos
    steps = 200
    pick_traj = [
        ((1 - t) * init_qpos + t * target_qpos).tolist()
        for t in np.linspace(0, 1, steps)
    ]
    pick_back_traj = pick_traj[::-1]

    place_traj = [q.copy() for q in pick_traj]
    for q in place_traj:
        q[0] += np.deg2rad(30)

    # Genera init_qpos to place_traj[0] trajectory
    init_to_place_traj = [
        ((1 - t) * init_qpos + t * np.array(place_traj[0])).tolist()
        for t in np.linspace(0, 1, steps)
    ]
    place_back_traj = [q.copy() for q in place_traj[::-1]]
    time.sleep(1)

    move_id, end_id = supervisior.move_joint_trajectory_async(pick_traj, interval=0.005)
    res_id = supervisior.call_srv_async(
        "gripper", {"action": "close"}, MoveSyncOption.sync_w_latest_move()
    )
    move_id, end_id = supervisior.move_joint_trajectory_async(
        pick_back_traj, interval=0.0025
    )
    move_id, end_id = supervisior.move_joint_trajectory_async(
        init_to_place_traj, interval=0.0025
    )
    move_id, end_id = supervisior.move_joint_trajectory_async(
        place_traj, interval=0.0025
    )
    res_id = supervisior.call_srv_async(
        "gripper", {"action": "open"}, MoveSyncOption.sync_w_latest_move()
    )
    supervisior.wait_move()

    # Wait for execution to complete
    time.sleep(0.5)

    # Query final state
    state = task.query_state()
    print(f"Final robot state: {state}")

    # Stop the task
    supervisior.stop()
