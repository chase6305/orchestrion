from .generic_task import GenericTask
from .function_call_task import (
    InPlaceFunctionCallTask,
    ThreadedPoolFunctionCallTask,
)
from .reduced_robot_task_interface import ReducedRobotTaskInterface
from .modular_reduced_robot_task import ModularReducedRobotTask, SubModuleTask

all = [
    "GenericTask",
    "InPlaceFunctionCallTask",
    "ThreadedPoolFunctionCallTask",
    "ReducedRobotTaskInterface",
    "ModularReducedRobotTask",
    "SubModuleTask",
]
