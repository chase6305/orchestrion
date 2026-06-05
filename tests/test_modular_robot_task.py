import time
import queue
from orchestrion.tasks.modular_reduced_robot_task import (
    ModularReducedRobotTask,
    SubModuleTask,
)


def test_modular_robot_task_basic():
    print("[Test] Creating submodules...")
    sub1 = SubModuleTask()
    sub1.name = "arm"
    sub1.dof_begin = 0
    sub1.dof_end = 3
    sub2 = SubModuleTask()
    sub2.name = "gripper"
    sub2.dof_begin = 3
    sub2.dof_end = 4
    submodules = [sub1, sub2]

    print(f"[Test] Submodules: {[sub.name for sub in submodules]}")

    # Initial joint positions
    init_q = [0.0, 0.0, 0.0, 0.0]
    print(f"[Test] Initial joint positions: {init_q}")
    robot_task = ModularReducedRobotTask(init_q, submodules)
    robot_task.initialize()
    print("[Test] ModularReducedRobotTask initialized.")

    # Test move_sub_module for arm
    arm_traj = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    print(f"[Test] Sending arm trajectory: {arm_traj}")
    move_id = robot_task.move_sub_module(
        "arm", [q + [0.0] for q in arm_traj], interval=0.01
    )
    print(f"[Test] move_id for arm: {move_id}")
    assert move_id >= 0

    # Test move_sub_module for gripper
    grip_traj = [[0.0, 0.0, 0.0, 0.7], [0.0, 0.0, 0.0, 0.8]]
    print(f"[Test] Sending gripper trajectory: {grip_traj}")
    move_id2 = robot_task.move_sub_module("gripper", grip_traj, interval=0.01)
    print(f"[Test] move_id for gripper: {move_id2}")
    assert move_id2 >= 0

    # Allow some time for background thread to process
    time.sleep(0.1)

    # Print submodule queues (may be empty if consumed)
    for sub in submodules:
        items = []
        try:
            while True:
                items.append(sub.task_queue.get_nowait())
        except queue.Empty:
            pass
        print(f"[Test] Submodule '{sub.name}' task_queue contents after run: {items}")

    # Print current state
    state = robot_task.query_state()
    print(f"[Test] ModularReducedRobotTask state: {state}")

    robot_task.stop()
    print("[Test] ModularReducedRobotTask stopped.")
    print("test_modular_robot_task_basic passed.")


def test_modular_robot_task_basic_1():
    """
    Basic test for ModularReducedRobotTask:
    - Initialization
    - Valid trajectory
    - State query
    - Submodule queue update
    - Stop
    """

    def make_submodules():
        # Create two submodules, each controlling 2 DOF
        return [SubModuleTask(), SubModuleTask()]

    submodules = make_submodules()
    submodules[0].name = "arm"
    submodules[0].dof_begin = 0
    submodules[0].dof_end = 6
    submodules[1].name = "gripper"
    submodules[1].dof_begin = 6
    submodules[1].dof_end = 8
    init_q = [0.0] * 8
    robot_task = ModularReducedRobotTask(init_q, submodules)
    robot_task.initialize()

    # Valid trajectory: 2 steps
    traj = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 8.0, 9.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 8.0, 9.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 8.0, 9.0],
    ]
    move_id = robot_task.move_joint_trajectory_async(traj, interval=0.05)
    assert move_id == 0
    print(f"[ModularReducedRobotTask] Valid input test passed, move_id={move_id}.")

    gripper_traj = [[1.0, 2.0], [3.0, 4.0]]

    print(f"[Debug] gripper_traj: {gripper_traj}")
    move_id = robot_task.move_sub_module(
        submodule_name="gripper", motion_target=gripper_traj, interval=0.05
    )
    print(f"[Debug] move_id for gripper: {move_id}")
    time.sleep(0.2)
    state = robot_task.query_state()
    print(f"[Debug] State after gripper move: {state}")

    # Check gripper submodule queue
    gripper_sub = None
    for sub in submodules:
        if sub.name == "gripper":
            gripper_sub = sub
            break
    if gripper_sub:
        items = []
        while not gripper_sub.task_queue.empty():
            items.append(gripper_sub.task_queue.get())
        print(f"[Debug] gripper task_queue: {items}")
        # assert that items contain the expected trajectory commands
        assert len(items) > 0, f"Submodule gripper queue is empty!"
        assert any(isinstance(i, tuple) or hasattr(i, "motion_target") for i in items)

    # Wait for execution
    time.sleep(0.3)
    state = robot_task.query_state()
    print(f"[ModularReducedRobotTask] State after trajectory: {state}")
    assert state.latest_sent_id == 1
    assert state.latest_finished_id >= -1
    assert len(state.jps) == 8

    # Debug: print submodule config
    for sub in submodules:
        print(
            f"Submodule config: name={sub.name}, dof_begin={sub.dof_begin}, dof_end={sub.dof_end}"
        )

    # Debug: print submodule queues
    for sub in submodules:
        print(f"Submodule {sub.name} received: (skip check, already consumed above)")

    # Stop
    robot_task.stop()
    print("[ModularReducedRobotTask] Stop test passed.")


def test_modular_robot_task_left_right_arm():
    """
    Test ModularReducedRobotTask with left_arm and right_arm, each with 6 DOF.
    """
    left_arm = SubModuleTask()
    right_arm = SubModuleTask()
    left_arm.name = "left_arm"
    left_arm.dof_begin = 0
    left_arm.dof_end = 6
    right_arm.name = "right_arm"
    right_arm.dof_begin = 6
    right_arm.dof_end = 12
    submodules = [left_arm, right_arm]

    init_q = [0.0] * 12
    robot_task = ModularReducedRobotTask(init_q, submodules)
    robot_task.initialize()

    traj = [[i + 1.0 for i in range(12)], [i + 2.0 for i in range(12)]]
    move_id = robot_task.move_joint_trajectory_async(traj, interval=0.05)
    assert move_id == 0

    time.sleep(0.2)
    state = robot_task.query_state()
    print(f"[ModularReducedRobotTask] State after trajectory: {state}")
    assert state.latest_sent_id == 0
    assert state.latest_finished_id >= -1
    assert len(state.jps) == 12

    for sub in submodules:
        print(
            f"Submodule config: name={sub.name}, dof_begin={sub.dof_begin}, dof_end={sub.dof_end}"
        )
        items = []
        while not sub.task_queue.empty():
            items.append(sub.task_queue.get())
        print(f"Submodule {sub.name} received: {items}")

    robot_task.stop()
    print("[ModularReducedRobotTask] Stop test passed.")


if __name__ == "__main__":
    test_modular_robot_task_basic()
    test_modular_robot_task_basic_1()
    test_modular_robot_task_left_right_arm()
