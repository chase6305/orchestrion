"""Native wait bounds preserve pending work and representable positive cadence."""

import threading
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import pytest

from orchestrion import (
    CallableTask,
    MoveSyncOption,
    RetryPolicy,
    SimulatedRobotTask,
    TaskPilot,
)
from orchestrion.tasks import (
    InPlaceFunctionCallTask,
    ModularReducedRobotTask,
    PollingTask,
    SubModuleTask,
)
from orchestrion.utils.types import PeekResponseResultType

_ROUNDS_TO_LIMIT = Fraction(threading.TIMEOUT_MAX) + Fraction(1, 2**100)
_ROUNDS_TO_ZERO = Fraction(1, 10**400)
_EXCESSIVE_WAITS = [
    pytest.param(1e300, id="large-float"),
    pytest.param(_ROUNDS_TO_LIMIT, id="fraction-rounds-to-limit"),
]


class ModularRobot(ModularReducedRobotTask):
    def _on_initial_state_bg_thread(self, joint_positions):
        pass

    def _on_robot_state_bg_thread(self, joint_positions, move_id):
        pass


def _modular_robot(interval=0.001):
    return ModularRobot(
        [0.0, 0.0, 0.0],
        SubModuleTask("arm", 0, 1),
        [SubModuleTask("left", 1, 1), SubModuleTask("right", 2, 1)],
        interval=interval,
    )


@pytest.fixture(params=["simulated", "modular"])
def robot(request):
    instance = (
        SimulatedRobotTask([0.0]) if request.param == "simulated" else _modular_robot()
    )
    try:
        yield instance
    finally:
        instance.stop(timeout=1)


@pytest.mark.parametrize(
    "interval", _EXCESSIVE_WAITS + [pytest.param(_ROUNDS_TO_ZERO, id="underflow")]
)
@pytest.mark.parametrize(
    "factory",
    [
        lambda interval: PollingTask(lambda: {}, interval=interval),
        lambda interval: TaskPilot(SimulatedRobotTask([0.0]), poll_interval=interval),
        _modular_robot,
    ],
    ids=["polling-task", "task-pilot", "robot-scheduler"],
)
def test_background_cadence_rejects_unrepresentable_waits(factory, interval):
    if interval > threading.TIMEOUT_MAX and isinstance(interval, Fraction):
        assert float(interval) == threading.TIMEOUT_MAX
    message = "TIMEOUT_MAX" if interval > threading.TIMEOUT_MAX else "positive"
    with pytest.raises(ValueError, match=message):
        factory(interval)


def test_wait_move_rejects_intervals_that_round_to_zero(robot):
    with pytest.raises(ValueError, match="positive"):
        robot.wait_move(time_out=0, interval=_ROUNDS_TO_ZERO)


@pytest.mark.parametrize("interval", [Fraction(1, 1000), 5e-324])
def test_representable_positive_trajectory_intervals_still_queue(robot, interval):
    assert robot.move_joint_trajectory_async([[0.1]], interval=interval) == 0
    assert robot.query_state().latest_sent_id == 0


@pytest.mark.parametrize("timeout", _EXCESSIVE_WAITS)
def test_invalid_robot_stop_preserves_pending_moves_and_admission(robot, timeout):
    assert robot.move_joint_trajectory_async([[0.1]], interval=0.001) == 0
    before = robot.query_state()
    with pytest.raises(ValueError, match="TIMEOUT_MAX"):
        robot.stop(timeout=timeout)
    assert robot.query_state() == before
    assert robot.move_joint_trajectory_async([[0.2]], interval=0.001) == 1
    robot.initialize()
    assert robot.wait_move(time_out=1)
    assert robot.query_state().cancelled_move_ids == ()


@pytest.mark.parametrize("timeout", _EXCESSIVE_WAITS)
def test_invalid_pilot_stop_preserves_pending_requests_and_admission(timeout):
    class Echo(InPlaceFunctionCallTask):
        def _call_fn(self, request_id, content):
            return content

    with TaskPilot(SimulatedRobotTask([0.0]), {"echo": Echo()}) as pilot:
        pending = pilot.call_srv_async("echo", {}, MoveSyncOption.sync_w_explicit_id(5))
        with pytest.raises(ValueError, match="TIMEOUT_MAX"):
            pilot.stop(timeout=timeout)
        assert pilot.is_running
        assert not pilot.query_request(pending).status.terminal
        immediate = pilot.call_srv_async(
            "echo", {"alive": True}, MoveSyncOption.no_sync()
        )
        assert pilot.wait_request(immediate, timeout=1).content == {"alive": True}
        assert not pilot.query_request(pending).status.terminal


@pytest.mark.parametrize(
    "interval", _EXCESSIVE_WAITS + [pytest.param(_ROUNDS_TO_ZERO, id="underflow")]
)
def test_invalid_trajectory_does_not_consume_id_and_exact_limit_can_queue(
    robot, interval
):
    assert robot.move_joint_trajectory_async([[0.1]], interval=interval) == -1
    assert robot.query_state().latest_sent_id == -1
    assert (
        robot.move_joint_trajectory_async([[0.2]], interval=threading.TIMEOUT_MAX) == 0
    )
    # No worker starts, so accepting the exact boundary never waits for its cadence.
    robot.stop(timeout=threading.TIMEOUT_MAX)
    assert robot.query_state().cancelled_move_ids == (0,)


@pytest.mark.parametrize(
    "interval", _EXCESSIVE_WAITS + [pytest.param(_ROUNDS_TO_ZERO, id="underflow")]
)
def test_submodule_cadence_rejection_preserves_single_and_batch_ids(
    interval,
):
    robot = _modular_robot()
    try:
        assert robot.move_submodule_trajectory_async("left", [[0.1]], interval) == -1
        assert not robot.move_submodule_async("left", [[0.1]], interval)
        assert (
            robot.move_submodules_trajectories_async(
                {"left": [[0.1]], "right": [[0.2]]}, interval
            )
            == {}
        )
        assert all(
            state.latest_sent_id == -1
            for state in robot.query_submodule_states().values()
        )
        assert robot.move_submodules_trajectories_async(
            {"left": [[0.1]], "right": [[0.2]]}, threading.TIMEOUT_MAX
        ) == {"left": 0, "right": 0}
        assert (
            robot.move_submodule_trajectory_async(
                "left", [[0.3]], threading.TIMEOUT_MAX
            )
            == 1
        )
        robot.stop(timeout=threading.TIMEOUT_MAX)
        assert robot.query_submodule_state("left").cancelled_move_ids == (0, 1)
        assert robot.query_submodule_state("right").cancelled_move_ids == (0,)
    finally:
        robot.stop(timeout=1)


@pytest.mark.parametrize("delay", _EXCESSIVE_WAITS)
def test_retry_policy_allows_large_math_but_actual_wait_rejects_overflow(delay):
    policy = RetryPolicy(max_attempts=2, delay=delay, max_delay=delay)
    assert policy.delay_before(2) == delay
    attempts = []
    completed = threading.Event()

    def unavailable(request_id, content):
        attempts.append(request_id)
        raise ConnectionError("offline")

    task = CallableTask(unavailable, retry_policy=policy)

    def notify_finished():
        if task.peek_status()["pending"] == 0:
            completed.set()

    task.set_completion_callback(notify_finished)
    with ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0)
            assert completed.wait(1)
            response = task.peek_response(0)
            assert response.result_type is PeekResponseResultType.ErrorUnknown
            assert "retry delay" in response.error and "TIMEOUT_MAX" in response.error
            assert attempts == [0]
        finally:
            task.stop()


def test_already_requested_stop_wins_over_excessive_retry_delay():
    entered, release = threading.Event(), threading.Event()

    def unavailable(request_id, content):
        entered.set()
        assert release.wait(2)
        raise ConnectionError("offline")

    task = CallableTask(
        unavailable,
        retry_policy=RetryPolicy(max_attempts=2, delay=1e300, max_delay=1e300),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        task.initialize(executor=executor)
        try:
            assert task.invoke_async(0)
            assert entered.wait(1)
            task.stop()
        finally:
            release.set()
            task.stop()
        outcome = task._future_map[0].result(timeout=1)
        assert outcome.cancelled
        assert "stopped" in outcome.error
