"""Select the main or submodule motion a peripheral request waits for."""

from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union


def _validation_error(
    need_sync: bool, associated_move_id: int, submodule_name: Optional[str]
) -> Optional[Tuple[Type[Exception], str]]:
    """Share error facts without raising during asynchronous preparation."""
    if not isinstance(need_sync, bool):
        return TypeError, "need_sync must be a bool"
    if isinstance(associated_move_id, bool) or not isinstance(associated_move_id, int):
        return TypeError, "associated_move_id must be an integer"
    if associated_move_id < -1:
        return ValueError, "associated_move_id must be -1 or non-negative"
    if not need_sync and associated_move_id != -1:
        return ValueError, "no-sync options cannot specify a move ID"
    if submodule_name is not None and (
        not isinstance(submodule_name, str) or not submodule_name
    ):
        return ValueError, "submodule_name must be a non-empty string or None"
    if not need_sync and submodule_name is not None:
        return ValueError, "no-sync options cannot specify a submodule"
    return None


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
        error = _validation_error(
            self.need_sync, self.associated_move_id, self.submodule_name
        )
        if error is not None:
            error_type, message = error
            raise error_type(message)

    def _snapshot(self) -> Union["MoveSyncOption", str]:
        """Return validated plain values or an error, preserving mutable builders."""
        need_sync = self.need_sync
        move_id = self.associated_move_id
        submodule_name = self.submodule_name
        if isinstance(move_id, int) and not isinstance(move_id, bool):
            move_id = int.__int__(move_id)
        if isinstance(submodule_name, str):
            submodule_name = str.__str__(submodule_name)
        error = _validation_error(need_sync, move_id, submodule_name)
        if error is not None:
            return error[1]
        return MoveSyncOption(need_sync, move_id, submodule_name)

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
