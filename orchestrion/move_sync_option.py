from dataclasses import dataclass


@dataclass
class MoveSyncOption(object):
    """
    Option class for movement synchronization.

    Attributes:
        need_sync (bool): Whether synchronization is required.
        associated_move_id (int): The move ID to synchronize with.
                                  -1 means latest.
    """

    need_sync: bool = True
    associated_move_id: int = -1

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

    @staticmethod
    def sync_w_latest_move() -> "MoveSyncOption":
        """
        Create a MoveSyncOption to sync with the latest move.

        Returns:
            MoveSyncOption: Option to sync with the latest move.
        """
        return MoveSyncOption(need_sync=True, associated_move_id=-1)

    @staticmethod
    def no_sync() -> "MoveSyncOption":
        """
        Create a MoveSyncOption with no synchronization.

        Returns:
            MoveSyncOption: Option with no synchronization.
        """
        return MoveSyncOption(need_sync=False, associated_move_id=-1)

    @staticmethod
    def sync_w_explicit_id(move_id: int) -> "MoveSyncOption":
        """
        Create a MoveSyncOption to sync with a specific move ID.

        Args:
            move_id (int): The explicit move ID to sync with.

        Returns:
            MoveSyncOption: Option to sync with the given move ID.
        """
        return MoveSyncOption(need_sync=True, associated_move_id=move_id)
