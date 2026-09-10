"""Retained outcome statistics across completion, cancellation, and pruning."""

import concurrent.futures

from orchestrion.tasks import ThreadedPoolFunctionCallTask


class Task(ThreadedPoolFunctionCallTask):
    def _call_fn(self, request_id, content):
        return content


class Executor:
    def __init__(self, future_type=concurrent.futures.Future):
        self.future_type = future_type
        self.futures = []

    def submit(self, call, *args):
        future = self.future_type()
        self.futures.append(future)
        return future


def counts(task):
    status = task.peek_status()
    values = tuple(status[key] for key in ("pending", "succeeded", "failed", "cancelled"))
    assert sum(values) == status["retained_results"]
    return values


def test_status_tracks_eviction_pruning_and_shutdown():
    executor = Executor()
    task = Task(max_result_count=2)
    task.initialize(executor=executor)
    try:
        for request_id in range(5):
            assert task.invoke_async(request_id)
        assert counts(task) == (5, 0, 0, 0)
        executor.futures[0].set_result(None)
        executor.futures[1].set_exception(ValueError("failed"))
        assert counts(task) == (3, 1, 1, 0)
        assert task.cancel_request(2)
        assert counts(task) == (2, 0, 1, 1)
        assert task.forget_response(1)
        assert counts(task) == (2, 0, 0, 1)
        executor.futures[3].set_result(None)
        assert counts(task) == (1, 1, 0, 1)
        task.stop()
        assert counts(task) == (0, 1, 0, 1)
        assert not task.peek_status()["initialized"]
        assert task.forget_response(3)
        assert task.forget_response(4)
        assert counts(task) == (0, 0, 0, 0)
    finally:
        task.stop()


def test_delayed_callbacks_do_not_leave_stale_or_duplicate_counts():
    class DelayedFuture(concurrent.futures.Future):
        def add_done_callback(self, callback):
            self.callback = callback

    executor = Executor(DelayedFuture)
    task = Task(max_result_count=1)
    task.initialize(executor=executor)
    try:
        assert task.invoke_async(0)
        future = executor.futures[0]
        future.set_result(None)
        assert counts(task) == (0, 1, 0, 0)
        assert task.forget_response(0)
        future.callback(future)
        assert counts(task) == (0, 0, 0, 0)
        assert task.invoke_async(1)
        future = executor.futures[1]
        future.set_exception(ValueError("failed"))
        assert counts(task) == (0, 0, 1, 0)
        future.callback(future)
        future.callback(future)
        assert counts(task) == (0, 0, 1, 0)
    finally:
        task.stop()


def test_status_does_not_reinspect_completed_history():
    class ObservedFuture(concurrent.futures.Future):
        reads = 0

        def done(self):
            self.reads += 1
            return super().done()

        def cancelled(self):
            self.reads += 1
            return super().cancelled()

        def exception(self, timeout=None):
            self.reads += 1
            return super().exception(timeout)

    executor = Executor(ObservedFuture)
    task = Task()
    task.initialize(executor=executor)
    try:
        for request_id in range(100):
            assert task.invoke_async(request_id)
            executor.futures[-1].set_result(None)
        reads = sum(future.reads for future in executor.futures)
        for _ in range(10):
            assert counts(task) == (0, 100, 0, 0)
        assert sum(future.reads for future in executor.futures) == reads
    finally:
        task.stop()


def test_callable_cancellation_exception_is_distinct_from_cancelled_future():
    executor = Executor()
    task = Task()
    task.initialize(executor=executor)
    try:
        assert task.invoke_async(0)
        assert task.invoke_async(1)
        executor.futures[0].set_exception(concurrent.futures.CancelledError("SDK stopped"))
        assert executor.futures[1].cancel()
        assert counts(task) == (0, 0, 1, 1)
        assert task.forget_response(0)
        assert counts(task) == (0, 0, 0, 1)
    finally:
        task.stop()
