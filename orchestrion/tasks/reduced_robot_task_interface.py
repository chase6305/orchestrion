import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import List, Optional


class ReducedRobotTaskInterface(ABC):

    @dataclass
    class ReducedRobotState(object):
        """
        Data class representing the robot's joint positions and movement IDs.
        """

        jps: List[float]  # Joint positions
        latest_finished_id: int = -1  # Last finished movement ID
        latest_sent_id: int = -1  # Last sent movement ID

    @dataclass
    class MovementRequest(object):
        """
        Data class representing a movement request for the robot.
        """

        motion_target: List[List[float]]  # List of joint positions for the trajectory
        interval: float = 0.01  # Time interval between points
        motion_id_begin: int = 0  # Starting movement ID
        endpoint_index: Optional[List[int]] = None  # Segment endpoints

    @abstractmethod
    def initialize(self):
        """
        Initialize the robot state. Can be extended to read real robot state.
        Returns:
            bool: True if initialization succeeded.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """
        Stop the reduced robot task, clear movement queue, and invoke cancellation callbacks.
        Returns:
            bool: True if stopped successfully.
        """
        raise NotImplementedError

    @abstractmethod
    def move_joint_trajectory_async(
        self,
        motion_target: List[List[float]],
        interval: float = 0.01,
        endpoint_index: Optional[List[int]] = None,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def query_state(self) -> Optional[ReducedRobotState]:
        """
        Query the current robot state in a thread-safe manner.
        Returns:
            Optional[ReducedRobotState]: A copy of the current state.
        """
        raise NotImplementedError

    def full_task(self):
        """
        Return self if a full-featured task is needed.
        Returns:
            ReducedRobotTaskInterface: The current instance.
        """
        return self

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
            return False

        # Check each endpoint index is greater than previous
        prev_endpoint = 0
        n_endpoints = len(endpoint_index)
        for i in range(n_endpoints):
            endpoint_i = endpoint_index[i]
            if isinstance(endpoint_i, bool) or not isinstance(endpoint_i, int):
                return False
            if endpoint_i <= prev_endpoint:
                return False
            prev_endpoint = endpoint_i

        # Last endpoint must match the length of motion_target
        if prev_endpoint != len(motion_target):
            return False

        return True

    @staticmethod
    def _is_finite_trajectory(motion_target: List[List[float]]) -> bool:
        return all(
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(float(value))
            for point in motion_target
            for value in point
        )

    @abstractmethod
    def wait_move(self, time_out: float = -1, interval: float = 0.05) -> bool:
        raise NotImplementedError
