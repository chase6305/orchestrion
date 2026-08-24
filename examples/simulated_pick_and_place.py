"""Run a synchronized workflow without robot hardware."""

from orchestrion import MoveSyncOption, SimulatedRobotTask, TaskPilot
from orchestrion.tasks import InPlaceFunctionCallTask


class GripperTask(InPlaceFunctionCallTask):
    def _call_fn(self, request_id, content):
        return {"applied": content["action"]}


robot = SimulatedRobotTask([0.0, 0.0])
pilot = TaskPilot(robot, {"gripper": GripperTask()})
pilot.initialize()

try:
    _, move_end = pilot.move_joint_trajectory_async(
        [[0.2, 0.1], [0.8, 0.4]], interval=0.05
    )
    request_id = pilot.call_srv_async(
        "gripper",
        {"action": "close"},
        MoveSyncOption.sync_w_explicit_id(move_end - 1),
    )
    result = pilot.wait_request(request_id, timeout=2.0)
    print(result)
    for event in pilot.timeline(request_id):
        print(event)
finally:
    pilot.stop()
