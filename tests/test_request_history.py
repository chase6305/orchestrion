"""Exact deduplication must survive gaps, block boundaries, and result eviction."""

import random

import pytest

from orchestrion.utils.request_history import RequestHistory


@pytest.mark.parametrize("start, step", [(0, 1), (1, 1), (0, 2), (1000000, 1)])
def test_history_compresses_nearby_ids_without_filling_gaps(start, step):
    history = RequestHistory()
    accepted = set(range(start, start + 10000, step))
    for request_id in accepted:
        assert history.add(request_id)
    for request_id in range(start, start + 10001):
        assert (request_id in history) == (request_id in accepted)
    assert len(history._blocks) <= 41
    for request_id in accepted:
        assert not history.add(request_id)


def test_history_matches_set_with_random_order_and_duplicates():
    history = RequestHistory()
    accepted = set()
    rng = random.Random(42)
    for _ in range(10000):
        request_id = rng.randrange(2048)
        assert history.add(request_id) == (request_id not in accepted)
        accepted.add(request_id)
        probe = rng.randrange(2048)
        assert (probe in history) == (probe in accepted)
    for request_id in range(2048):
        history.add(request_id)
    assert history._blocks == {}
    assert all(request_id in history for request_id in range(2048))
    assert 2048 not in history


def test_history_supports_arbitrarily_large_ids():
    history = RequestHistory()
    request_id = 10 ** 100
    assert history.add(request_id)
    assert request_id in history
    assert request_id - 1 not in history
    assert request_id + 1 not in history
    assert not history.add(request_id)
