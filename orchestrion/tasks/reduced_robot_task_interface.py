from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from orchestrion.utils.numeric import is_finite_real


class ReducedRobotTaskInterface(ABC):
    @dataclass
    class ReducedRobotState:
        """Joint positions and main-movement completion/cancellation history.

        Cancelled IDs remain cancelled when later movements finish.
        """

        jps: List[float]  # Joint positions.
        latest_finished_id: int = -1  # No completed move when -1.
        latest_sent_id: int = -1  # No accepted move when -1.
        cancelled_move_ids: tuple = ()  # Accepted moves discarded before completion.

    @dataclass
    class MovementRequest:
        """A joint trajectory split into completion-aware segments."""

        motion_target: List[List[float]]
        interval: float = 0.01  # Seconds between trajectory points.
        motion_id_begin: int = 0  # First segment's move ID.
        endpoint_index: Optional[List[int]] = None  # Exclusive segment endpoints.

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

        previous_endpoint = 0
        for endpoint in endpoint_index:
            if isinstance(endpoint, bool) or not isinstance(endpoint, int):
                return False
            if endpoint <= previous_endpoint:
                return False
            previous_endpoint = endpoint

        return previous_endpoint == len(motion_target)

    # Preserve the validator name used by robot subclasses without another call.
    _is_finite_real = staticmethod(is_finite_real)

    @staticmethod
    def _is_finite_trajectory(motion_target: List[List[float]]) -> bool:
        return isinstance(motion_target, (list, tuple)) and all(
            isinstance(point, (list, tuple))
            and all(ReducedRobotTaskInterface._is_finite_real(value) for value in point)
            for point in motion_target
        )

    @abstractmethod
    def wait_move(self, time_out: float = -1, interval: float = 0.05) -> bool:
        raise NotImplementedError
