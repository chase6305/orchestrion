import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from orchestrion.utils.logger import logger


class ReducedRobotTaskInterface(object):

    @dataclass
    class ReducedRobotState(object):
        """
        Data class representing the robot's joint positions and movement IDs.
        """

        jps: List[float]  # Joint positions
        latest_finished_id: int = -1  # Last finished movement ID
        latest_sent_id: int = 1  # Last sent movement ID

    @dataclass
    class MovementRequest(object):
        """
        Data class representing a movement request for the robot.
        """

        motion_target: List[List[float]]  # List of joint positions for the trajectory
        interval: float = 0.01  # Time interval between points
        motion_id_begin: int = 0  # Starting movement ID
        endpoint_index: Optional[List[int]] = None  # Segment endpoints

    def __init__(self):
        """
        Initialize the reduced robot task, including state, movement queue, and thread safety.
        """
        pass

    def initialize(self):
        """
        Initialize the robot state. Can be extended to read real robot state.
        Returns:
            bool: True if initialization succeeded.
        """
        pass

    def stop(self):
        """
        Stop the reduced robot task, clear movement queue, and invoke cancellation callbacks.
        Returns:
            bool: True if stopped successfully.
        """
        pass

    def move_joint_trajectory_async(
        self,
        motion_target: List[List[float]],
        interval: float = 0.01,
        endpoint_index: Optional[List[int]] = None,
    ) -> int:
        return -1

    def query_state(self) -> Optional[ReducedRobotState]:
        """
        Query the current robot state in a thread-safe manner.
        Returns:
            Optional[ReducedRobotState]: A copy of the current state.
        """
        with self._lock:
            return copy.deepcopy(self._state)

    def full_task(self):
        """
        Return self if a full-featured task is needed.
        Returns:
            ReducedRobotTaskInterface: The current instance.
        """
        return None

    @staticmethod
    def _check_assigned_endpoint_index(
        motion_target: List[List[float]], endpoint_index: List[int]
    ) -> bool:
        """
        Check if the assigned endpoint indices are valid for the given motion target.
        Args:
            motion_target (List[List[float]]): List of joint positions.
            endpoint_index (List[int]): List of endpoint indices.
        Returns:
            bool: True if valid, False otherwise.
        """
        if endpoint_index is None or len(endpoint_index) == 0:
            return True

        # Check each endpoint index is greater than previous
        prev_endpoint = 0
        n_endpoints = len(endpoint_index)
        for i in range(n_endpoints):
            endpoint_i = endpoint_index[i]
            if endpoint_i <= prev_endpoint:
                return False
            prev_endpoint = endpoint_i

        # Last endpoint must match the length of motion_target
        if prev_endpoint != len(motion_target):
            return False

        return True

    def wait_move(self, time_out: float = -1, interval: float = 0.05) -> bool:
        return False
