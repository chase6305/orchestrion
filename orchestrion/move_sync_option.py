"""Select the main or submodule motion a peripheral request waits for."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MoveSyncOption:
    """Choose a motion dependency, resolved when the request is submitted.

    With synchronization enabled, move ID -1 selects the latest submitted move.
    ``submodule_name=None`` selects the main robot's movement history.
    """

    need_sync: bool = True
    associated_move_id: int = -1
    submodule_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.need_sync, bool):
            raise TypeError("need_sync must be a bool")
        if isinstance(self.associated_move_id, bool) or not isinstance(
            self.associated_move_id, int
        ):
            raise TypeError("associated_move_id must be an integer")
        if self.associated_move_id < -1:
            raise ValueError("associated_move_id must be -1 or non-negative")
        if not self.need_sync and self.associated_move_id != -1:
            raise ValueError("no-sync options cannot specify a move ID")
        if self.submodule_name is not None and (
            not isinstance(self.submodule_name, str) or not self.submodule_name
        ):
            raise ValueError("submodule_name must be a non-empty string or None")
        if not self.need_sync and self.submodule_name is not None:
            raise ValueError("no-sync options cannot specify a submodule")

    @staticmethod
    def sync_w_latest_move() -> "MoveSyncOption":
        """Wait for the latest submitted main move."""
        return MoveSyncOption(need_sync=True, associated_move_id=-1)

    @staticmethod
    def no_sync() -> "MoveSyncOption":
        """Dispatch without waiting for motion."""
        return MoveSyncOption(need_sync=False, associated_move_id=-1)

    @staticmethod
    def sync_w_explicit_id(move_id: int) -> "MoveSyncOption":
        """Wait for the specified main move ID; -1 selects the latest."""
        return MoveSyncOption(need_sync=True, associated_move_id=move_id)

    @staticmethod
    def sync_w_submodule(submodule_name: str, move_id: int = -1) -> "MoveSyncOption":
        """Synchronize with a specific or latest move of one robot submodule."""
        return MoveSyncOption(
            need_sync=True,
            associated_move_id=move_id,
            submodule_name=submodule_name,
        )
