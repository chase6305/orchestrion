"""Viewer startup failures release resources acquired before context entry."""

import threading
from unittest.mock import Mock

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yourdfpy")
pytest.importorskip("viser")

from examples.dual_arm.common import DualArmRuntime
from examples.dual_arm.viser import DualArmViserRuntime
from examples.viser.common import ViserDemoRuntime


@pytest.fixture
def failed_monitor(monkeypatch):
    error = RuntimeError("viewer startup failed")
    monitor = Mock()
    monitor.start.side_effect = error
    monkeypatch.setattr(threading, "Thread", Mock(return_value=monitor))
    return monitor, error


@pytest.mark.parametrize("failure_at", ["initialization", "monitor"])
def test_single_viewer_startup_failure_closes_owned_resources(
    failed_monitor, failure_at,
):
    monitor, error = failed_monitor
    runtime = object.__new__(ViserDemoRuntime)
    runtime.pilot = Mock()
    runtime.executor = Mock()
    runtime.server = Mock()
    runtime._monitor_stop = threading.Event()
    runtime._monitor_thread = None
    cleanup = Mock(wraps=runtime.__exit__)
    runtime.__exit__ = cleanup
    if failure_at == "initialization":
        runtime.pilot.initialize.side_effect = error

    with pytest.raises(RuntimeError) as raised:
        runtime.__enter__()
    assert raised.value is error
    assert cleanup.call_args.args[:2] == (RuntimeError, error)
    runtime.pilot.stop.assert_called_once()
    runtime.executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    runtime.server.stop.assert_called_once()
    assert runtime._monitor_stop.is_set()
    monitor.join.assert_not_called()


@pytest.mark.parametrize("failure_at", ["initialization", "monitor"])
def test_dual_viewer_startup_failure_closes_transport_and_started_runtime(
    monkeypatch, failed_monitor, failure_at,
):
    monitor, error = failed_monitor
    runtime = object.__new__(DualArmViserRuntime)
    runtime.server = Mock()
    runtime._visual_stop = threading.Event()
    runtime._visual_thread = None
    initialize = Mock(side_effect=error if failure_at == "initialization" else None)
    cleanup = Mock()
    monkeypatch.setattr(DualArmRuntime, "__enter__", initialize)
    monkeypatch.setattr(DualArmRuntime, "__exit__", cleanup)

    with pytest.raises(RuntimeError) as raised:
        runtime.__enter__()
    assert raised.value is error
    runtime.server.stop.assert_called_once()
    if failure_at == "monitor":
        cleanup.assert_called_once()
        assert cleanup.call_args.args[:2] == (RuntimeError, error)
        assert runtime._visual_stop.is_set()
    else:
        cleanup.assert_not_called()
    monitor.join.assert_not_called()
