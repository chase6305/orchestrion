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

    @staticmethod
    def sync_w_latest_move():
        """
        Create a MoveSyncOption to sync with the latest move.

        Returns:
            MoveSyncOption: Option to sync with the latest move.
        """
        return MoveSyncOption(need_sync=True, associated_move_id=-1)

    @staticmethod
    def no_sync():
        """
        Create a MoveSyncOption with no synchronization.

        Returns:
            MoveSyncOption: Option with no synchronization.
        """
        return MoveSyncOption(need_sync=False, associated_move_id=-1)

    @staticmethod
    def sync_w_explicit_id(move_id: int):
        """
        Create a MoveSyncOption to sync with a specific move ID.

        Args:
            move_id (int): The explicit move ID to sync with.

        Returns:
            MoveSyncOption: Option to sync with the given move ID.
        """
        return MoveSyncOption(need_sync=True, associated_move_id=move_id)
