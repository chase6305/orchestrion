"""Explicit pruning must preserve retention order and bounded bookkeeping."""

import random

import pytest

from orchestrion.utils.result_retention import ResultRetention


@pytest.mark.parametrize("capacity", [1, 3, 100])
def test_retention_matches_reference_with_out_of_order_completions(capacity):
    retention = ResultRetention(capacity)
    expected = set()
    rng = random.Random(42)
    ids = list(range(1000))
    rng.shuffle(ids)
    for request_id in ids:
        if expected and rng.random() < 0.6:
            removed = rng.choice(sorted(expected))
            expected.remove(removed)
            retention.discard(removed)
        expected.add(request_id)
        expired = min(expected) if len(expected) > capacity else None
        if expired is not None:
            expected.remove(expired)
        assert retention.add(request_id) == expired
        assert retention._live == expected
        assert len(retention._heap) <= 2 * len(expected) + 64


def test_pruning_compacts_stale_entries_and_releases_empty_heap():
    retention = ResultRetention(1000)
    for request_id in range(1000):
        retention.add(request_id)
    for request_id in range(999, 0, -1):
        retention.discard(request_id)
        assert len(retention._heap) <= 2 * len(retention._live) + 64
    retention.discard(0)
    assert retention._heap == []
    assert not retention._live


def test_full_retention_skips_pruned_head_and_rejects_older_result():
    retention = ResultRetention(2)
    assert retention.add(10) is None
    assert retention.add(20) is None
    retention.discard(10)
    assert retention.add(30) is None
    assert retention.add(5) == 5
    assert retention._live == {20, 30}
    assert retention.add(40) == 20
    assert retention._live == {30, 40}
