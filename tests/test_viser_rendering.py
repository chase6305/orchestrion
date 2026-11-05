"""Arm kinematics reuse, failure recovery and concurrent frame application."""

import concurrent.futures
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yourdfpy")
pytest.importorskip("viser")

import numpy as np

from examples.dual_arm.viser import DualArmViserRuntime, ReplayFrame


@pytest.fixture
def render_runtime():
    runtime = object.__new__(DualArmViserRuntime)
    runtime._render_lock = threading.Lock()
    runtime._rendered_arm_poses = {}
    runtime._arm_handles = {side: Mock() for side in ("left", "right")}
    runtime._ee_position = Mock(
        side_effect=lambda side, configuration, offset: np.array(
            [1.0 if side == "left" else -1.0, 0.0, 1.0]
        )
    )
    runtime._gripper_indicators = {
        side: SimpleNamespace(color=None) for side in ("left", "right")
    }
    runtime._gripper_fingers = {
        side: [SimpleNamespace(), SimpleNamespace()] for side in ("left", "right")
    }
    runtime._payloads = {
        name: SimpleNamespace(visible=False)
        for name in ("table", "left", "both", "right", "output")
    }
    runtime._zone_indicator = SimpleNamespace(color=None)
    runtime._state_status = SimpleNamespace(content="")
    return runtime


def test_unchanged_arm_poses_still_update_grippers_ownership_and_progress(
    render_runtime,
):
    runtime = render_runtime
    frame = ReplayFrame(0.0, (0.0,) * 3, (0.0,) * 3, 0, 0, None, "table", "idle")
    runtime._render_frame(frame)
    runtime._render_frame(
        replace(
            frame, elapsed=0.5, left_gripper=1, payload_owner="left_arm",
            zone_owner="cycle", event="grasp",
        )
    )
    assert all(
        handle.update_cfg.call_count == 1 for handle in runtime._arm_handles.values()
    )
    assert runtime._ee_position.call_count == 2
    assert runtime._gripper_indicators["left"].color == (230, 145, 55)
    assert runtime._payloads["left"].visible
    assert not runtime._payloads["table"].visible
    assert "0.50s" in runtime._state_status.content
    assert "grasp" in runtime._state_status.content

    runtime._render_frame(replace(frame, left_arm=(0.1, 0.2, 0.3)), replaying=True)
    runtime._render_frame(frame, replaying=True)
    assert runtime._arm_handles["left"].update_cfg.call_count == 3
    assert runtime._arm_handles["right"].update_cfg.call_count == 1
    assert "Replay" in runtime._state_status.content


@pytest.mark.parametrize("failure_at", ["renderer", "kinematics"])
def test_failed_arm_updates_are_retried_before_reusing_a_pose(
    render_runtime, failure_at,
):
    runtime = render_runtime
    frame = ReplayFrame(0.0, (0.0,) * 3, (0.0,) * 3, 0, 0, None, "table", "idle")
    runtime._render_frame(frame)
    method = (
        runtime._arm_handles["left"].update_cfg
        if failure_at == "renderer" else runtime._ee_position
    )
    previous = method.side_effect
    method.side_effect = RuntimeError("pose update failed")
    with pytest.raises(RuntimeError, match="pose update"):
        runtime._render_frame(replace(frame, left_arm=(0.1, 0.2, 0.3)))
    method.side_effect = previous
    runtime._render_frame(frame)
    assert runtime._arm_handles["left"].update_cfg.call_count == 3


def test_concurrent_renderers_do_not_interleave_frame_updates(render_runtime):
    runtime = render_runtime
    entered, release, second_started = (
        threading.Event(), threading.Event(), threading.Event()
    )
    entry_lock = threading.Lock()
    entries = []

    def update(configuration):
        with entry_lock:
            entries.append(configuration)
            first = len(entries) == 1
        if first:
            entered.set()
            assert release.wait(2)

    runtime._arm_handles["left"].update_cfg.side_effect = update
    frame = ReplayFrame(0.0, (0.0,) * 3, (0.0,) * 3, 0, 0, None, "table", "first")

    def render_next():
        second_started.set()
        runtime._render_frame(replace(frame, left_arm=(0.1, 0.2, 0.3), event="next"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(runtime._render_frame, frame)
        try:
            assert entered.wait(1)
            second = executor.submit(render_next)
            assert second_started.wait(1)
            with pytest.raises(concurrent.futures.TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
        first.result(timeout=1)
        second.result(timeout=1)
    assert "next" in runtime._state_status.content


