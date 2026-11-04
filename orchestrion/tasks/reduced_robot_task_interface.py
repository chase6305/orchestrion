import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import List, Optional


class ReducedRobotTaskInterface(ABC):

    @dataclass
    class ReducedRobotState:
        """Joint positions and main-movement completion/cancellation history."""

        jps: List[float]  # Joint positions
        latest_finished_id: int = -1  # Last finished movement ID
        latest_sent_id: int = -1  # Last sent movement ID
        cancelled_move_ids: tuple = ()  # Accepted movements discarded on stop

    @dataclass
    class MovementRequest:
        """A joint trajectory split into completion-aware segments."""

        motion_target: List[List[float]]  # List of joint positions for the trajectory
        interval: float = 0.01  # Time interval between points
        motion_id_begin: int = 0  # Starting movement ID
        endpoint_index: Optional[List[int]] = None  # Segment endpoints

    @abstractmethod
    def initialize(self):
        """Acquire backend resources and initialize robot state."""
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """Stop motion processing and release backend resources."""
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
        """Return a thread-safe state snapshot, or None when unavailable."""
        raise NotImplementedError

    def full_task(self):
        """Return this backend for legacy full-task callers."""
        return self

    @staticmethod
    def _check_assigned_endpoint_index(
        motion_target: List[List[float]], endpoint_index: List[int]
    ) -> bool:
        """Require increasing integer endpoints ending at the trajectory length."""
        if not isinstance(endpoint_index, (list, tuple)) or not endpoint_index:
            return False

        prev_endpoint = 0
        for endpoint_i in endpoint_index:
            if isinstance(endpoint_i, bool) or not isinstance(endpoint_i, int):
                return False
            if endpoint_i <= prev_endpoint:
                return False
            prev_endpoint = endpoint_i

        return prev_endpoint == len(motion_target)

    @staticmethod
    def _is_finite_real(value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, Real):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    @staticmethod
    def _is_finite_trajectory(motion_target: List[List[float]]) -> bool:
        return (
            isinstance(motion_target, (list, tuple))
            and all(
                isinstance(point, (list, tuple))
                and all(
                    ReducedRobotTaskInterface._is_finite_real(value)
                    for value in point
                )
                for point in motion_target
            )
        )

    @abstractmethod
    def wait_move(self, time_out: float = -1, interval: float = 0.05) -> bool:
        raise NotImplementedError
