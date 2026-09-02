import concurrent.futures
import importlib

import pytest

from examples.dual_arm.__main__ import main as dual_arm_main
from examples.dual_arm.common import (
    LEFT_HANDOFF,
    RIGHT_HANDOFF,
    DualArmRuntime,
    SharedZoneCoordinator,
    interpolate,
)
from orchestrion import CallableTask, MoveSyncOption


def test_interpolate_reaches_target_without_mutating_start():
    start = [0.0, 1.0]
    trajectory = interpolate(start, [1.0, -1.0], steps=4)
    assert len(trajectory) == 4
    assert trajectory[-1] == [1.0, -1.0]
    assert start == [0.0, 1.0]


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_interpolate_rejects_invalid_steps(steps):
    with pytest.raises(ValueError, match="steps"):
        interpolate([0.0], [1.0], steps)


def test_dual_arm_runtime_moves_both_arms_and_grippers_in_parallel():
    with DualArmRuntime() as runtime:
        moves = runtime.move_arms(LEFT_HANDOFF, RIGHT_HANDOFF, steps=4)
        requests = [
            runtime.command_gripper("left", "close"),
            runtime.command_gripper("right", "close"),
        ]
        runtime.wait_arms(moves, timeout=0.5)
        runtime.pilot.wait_requests(requests, timeout=0.5)
        assert runtime.arm_positions() == {
            "left": pytest.approx(LEFT_HANDOFF),
            "right": pytest.approx(RIGHT_HANDOFF),
        }
        assert runtime.robot.query_submodule_state("left_gripper").positions == [1.0]
        assert runtime.robot.query_submodule_state("right_gripper").positions == [1.0]


def test_dual_arm_runtime_event_history_is_a_defensive_copy():
    with DualArmRuntime() as runtime:
        runtime.record("nested", details={"value": 1})
        events = runtime.events
        events[-1]["details"]["value"] = 99
        assert runtime.events[-1]["details"]["value"] == 1


def test_service_can_synchronize_with_submodule_move_completion():
    with DualArmRuntime() as runtime:
        moves = runtime.move_arms(
            LEFT_HANDOFF, RIGHT_HANDOFF, steps=20, interval=0.005
        )
        request_id = runtime.command_gripper(
            "left",
            "close",
            MoveSyncOption.sync_w_submodule("left_arm", moves["left_arm"]),
        )
        result = runtime.pilot.wait_request(request_id, timeout=1.0)
        assert result.status.value == "succeeded"
        assert result.associated_submodule == "left_arm"
        statuses = [event.status.value for event in runtime.pilot.timeline(request_id)]
        assert statuses == ["queued", "waiting_for_move", "running", "succeeded"]


def test_submodule_synchronized_service_is_cancelled_with_associated_move():
    with DualArmRuntime() as runtime:
        moves = runtime.move_arms(
            LEFT_HANDOFF, RIGHT_HANDOFF, steps=50, interval=0.01
        )
        request_id = runtime.command_gripper(
            "left",
            "close",
            MoveSyncOption.sync_w_submodule("left_arm", moves["left_arm"]),
        )
        assert runtime.robot.cancel_submodule_move("left_arm", moves["left_arm"])
        result = runtime.pilot.wait_request(request_id, timeout=1.0)
        assert result.status.value == "cancelled"
        assert "submodule move" in result.error


def test_submodule_sync_does_not_depend_on_main_robot_state_query():
    with DualArmRuntime() as runtime:
        moves = runtime.move_arms(
            LEFT_HANDOFF, RIGHT_HANDOFF, steps=10, interval=0.003
        )

        def unavailable_main_state():
            raise RuntimeError("main state unavailable")

        runtime.robot.query_state = unavailable_main_state
        request_id = runtime.command_gripper(
            "right",
            "close",
            MoveSyncOption.sync_w_submodule("right_arm", moves["right_arm"]),
        )
        assert runtime.pilot.wait_request(request_id, timeout=1.0).status.value == (
            "succeeded"
        )


def test_shared_zone_reservation_is_idempotent_for_same_cycle():
    zone = SharedZoneCoordinator()
    first = zone.command(0, {"action": "reserve", "cycle": "cycle-a"})
    duplicate = zone.command(1, {"action": "reserve", "cycle": "cycle-a"})
    assert first["revision"] == duplicate["revision"] == 1
    with pytest.raises(RuntimeError, match="cycle-a"):
        zone.command(2, {"action": "reserve", "cycle": "cycle-b"})


@pytest.mark.parametrize(
    "content",
    [
        {"action": "enter", "cycle": "cycle-a"},
        {"action": "reserve", "cycle": ""},
        {"action": "reserve", "cycle": 1},
    ],
)
def test_shared_zone_rejects_invalid_direct_commands(content):
    with pytest.raises(ValueError, match="shared-zone"):
        SharedZoneCoordinator().command(0, content)


def test_shared_zone_allows_exactly_one_of_two_concurrent_cycles():
    zone = SharedZoneCoordinator()

    def reserve(cycle):
        try:
            zone.command(0, {"action": "reserve", "cycle": cycle})
            return True
        except RuntimeError:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        accepted = list(executor.map(reserve, ["cycle-a", "cycle-b"]))
    assert sorted(accepted) == [False, True]
    assert zone.status()["owner"] in {"cycle-a", "cycle-b"}


def test_dual_arm_parallel_pick_workflow():
    summary = importlib.import_module("examples.dual_arm.01_parallel_pick").run()
    assert summary["workflow"] == "parallel_pick"
    assert summary["left_gripper"] == 1.0
    assert summary["right_gripper"] == 1.0
    assert summary["arms"]["left"] != summary["arms"]["right"]
    assert summary["health"]["healthy"]


def test_dual_arm_handoff_transfers_payload_and_releases_zone():
    summary = importlib.import_module("examples.dual_arm.02_handoff").run()
    assert summary["payload_owner"] == "right_arm"
    assert summary["left_gripper"] == 0.0
    assert summary["right_gripper"] == 1.0
    assert summary["zone"]["owner"] is None
    owners = [
        event["owner"]
        for event in summary["events"]
        if event["event"] == "payload_owner_changed"
    ]
    assert owners == ["left_arm", "both_arms", "right_arm"]


def test_dual_arm_shared_zone_workflow_rejects_conflict_then_recovers():
    summary = importlib.import_module("examples.dual_arm.03_shared_zone").run()
    assert summary["first_status"] == "succeeded"
    assert summary["blocked_status"] == "failed"
    assert "cycle-a" in summary["blocked_error"]
    assert summary["retry_status"] == "succeeded"
    assert summary["zone"]["owner"] is None


def test_dual_arm_coordinated_abort_cancels_peer_motion():
    summary = importlib.import_module("examples.dual_arm.04_coordinated_abort").run()
    assert summary["workflow"] == "coordinated_abort"
    assert summary["left_cancelled"]
    assert summary["right_cancelled"]
    assert "peer cancellation" in summary["error"]
    assert any(event["event"] == "arms_aborted" for event in summary["events"])


def test_dual_arm_health_monitor_observes_monotonic_revisions():
    summary = importlib.import_module("examples.dual_arm.06_health_monitor").run()
    revisions = summary["observed_revisions"]
    assert revisions
    assert revisions == sorted(set(revisions))
    assert summary["request_samples"][-1]["succeeded"] >= 1
    assert summary["health"]["requests"]["succeeded"] == 2


def test_dual_arm_handoff_and_place_completes_full_cycle():
    summary = importlib.import_module(
        "examples.dual_arm.07_handoff_and_place"
    ).run()
    assert summary["workflow"] == "handoff_and_place"
    assert summary["payload_owner"] == "output_table"
    assert summary["left_gripper"] == 0.0
    assert summary["right_gripper"] == 0.0
    assert summary["zone"]["owner"] is None
    assert summary["arms"]["left"] == pytest.approx([0.0, 0.25, -0.4])
    assert summary["arms"]["right"] == pytest.approx([0.0, -0.25, 0.4])


def test_dual_arm_launcher_lists_demos(capsys):
    dual_arm_main(["--list"])
    output = capsys.readouterr().out
    assert "01  Parallel Pick" in output
    assert "02  Payload Handoff" in output
    assert "03  Shared Zone Safety" in output
    assert "04  Coordinated Abort" in output
    assert "05  Remote Inspection" in output
    assert "06  Health Revision Monitor" in output
    assert "07  Handoff and Place" in output


def test_dual_arm_runtime_rejects_duplicate_extra_service_names():
    with pytest.raises(ValueError, match="duplicate runtime services"):
        DualArmRuntime(
            extra_services={
                "shared_zone": CallableTask(lambda request_id, content: {})
            }
        )


@pytest.mark.parametrize(
    "services",
    [
        [],
        {"bad": object()},
        {"": CallableTask(lambda request_id, content: {})},
    ],
)
def test_dual_arm_runtime_validates_extra_services(services):
    with pytest.raises(TypeError, match="extra_services"):
        DualArmRuntime(extra_services=services)


def test_dual_arm_runtime_closes_executor_when_initialization_fails():
    failing = CallableTask(lambda request_id, content: {})

    def fail_initialize(**kwargs):
        raise RuntimeError("adapter initialization failed")

    failing.initialize = fail_initialize
    runtime = DualArmRuntime(extra_services={"failing": failing})
    with pytest.raises(RuntimeError, match="adapter initialization failed"):
        runtime.__enter__()
    with pytest.raises(RuntimeError, match="shutdown"):
        runtime._executor.submit(lambda: None)
