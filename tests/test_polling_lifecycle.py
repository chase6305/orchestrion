"""Deterministic races between telemetry status reads and shutdown."""

import threading
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from orchestrion.tasks.polling_task import PollingTask


def test_fraction_interval_polls_repeatedly_and_stops_cleanly():
    polled_twice = threading.Event()
    samples = []

    def poll():
        samples.append(len(samples) + 1)
        if len(samples) >= 2:
            polled_twice.set()
        return {"value": samples[-1]}

    task = PollingTask(poll, interval=Fraction(1, 100))
    task.initialize()
    try:
        assert polled_twice.wait(1)
        assert task.peek_status()["health"] == "online"
    finally:
        task.stop()
    assert not task.peek_status()["initialized"]


def test_status_remains_readable_when_stop_clears_worker_reference():
    captured, resume = threading.Event(), threading.Event()

    class Task(PollingTask):
        def __getattribute__(self, name):
            value = super().__getattribute__(name)
            if (
                name == "_poll_thread"
                and threading.current_thread().name.startswith("status-reader")
                and not captured.is_set()
            ):
                # Pause the reader after taking its first worker reference.
                captured.set()
                assert resume.wait(2)
            return value

    task = Task(lambda: {}, interval=1)
    task.initialize()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="status-reader") as executor:
        status = executor.submit(task.peek_status)
        try:
            assert captured.wait(1)
            task.stop()
            assert task._poll_thread is None
        finally:
            resume.set()
            task.stop()
        result = status.result(1)
    assert result["health"] == "offline"
    assert not result["initialized"]
    assert not result["available"]


def test_worker_can_request_stop_while_external_stop_joins_it(monkeypatch):
    started, returned = threading.Event(), threading.Event()

    def poll():
        started.set()
        assert task._poll_stop.wait(2)
        task.stop()
        returned.set()
        return {}

    task = PollingTask(poll, interval=1)
    task.initialize()
    thread = task._poll_thread
    original_join = thread.join

    def join(timeout=None):
        # The worker must finish its own stop while the external lifecycle
        # lock is still held, rather than waiting for the join timeout.
        assert returned.wait(0.5)
        original_join(timeout)

    monkeypatch.setattr(thread, "join", join)
    try:
        assert started.wait(1)
        task.stop()
        assert not thread.is_alive()
        assert task._poll_thread is None
    finally:
        task._poll_stop.set()
        original_join(1)
