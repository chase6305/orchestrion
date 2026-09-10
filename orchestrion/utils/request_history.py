"""Compact exact request history. Callers provide synchronization and validation."""


class RequestHistory:
    """Pack nearby IDs into bitmaps and discard fully accepted leading blocks.

    Gaps remain distinguishable from accepted IDs after responses are evicted.
    A fixed block size also bounds shifts for arbitrarily large request IDs.
    """

    _BLOCK_SIZE = 256
    _FULL_BLOCK = (1 << _BLOCK_SIZE) - 1

    def __init__(self) -> None:
        self._floor_block = 0
        self._blocks = {}

    def __contains__(self, request_id: int) -> bool:
        block, offset = divmod(request_id, self._BLOCK_SIZE)
        return block < self._floor_block or bool(
            self._blocks.get(block, 0) & (1 << offset)
        )

    def add(self, request_id: int) -> bool:
        """Return whether this ID was newly accepted."""
        block, offset = divmod(request_id, self._BLOCK_SIZE)
        if block < self._floor_block:
            return False
        previous = self._blocks.get(block, 0)
        bit = 1 << offset
        if previous & bit:
            return False
        updated = previous | bit
        self._blocks[block] = updated
        # Only filling the leading block can advance the watermark. Most adds
        # need no lookup of the leading block at all.
        if block == self._floor_block and updated == self._FULL_BLOCK:
            del self._blocks[block]
            self._floor_block += 1
            while self._blocks.get(self._floor_block) == self._FULL_BLOCK:
                del self._blocks[self._floor_block]
                self._floor_block += 1
        return True
