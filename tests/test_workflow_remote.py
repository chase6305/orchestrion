"""Ticket adapters distinguish transport acceptance from physical completion."""

import asyncio
from unittest.mock import patch

import pytest

from orchestrion.workflow import (
    Action,
    ExecutionUncertain,
    JournalError,
    Node,
    NodeStatus,
    OperationRejected,
    OperationStatus,
    OperationUpdate,
    Resource,
    ResourceClaim,
    RunStatus,
    SQLiteJournal,
    TicketAction,
    Workflow,
    WorkflowEngine,
    workflow_action,
)


class FakeBackend:
    def __init__(self, status=OperationStatus.SUCCEEDED):
        self.status = status
        self.submissions = []
        self.queries = 0
        self.cancels = 0
        self.submitted = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.query_effects = []
        self.submit_error = None
        self.cancel_error = None

    async def submit(self, operation_id, inputs):
        self.submissions.append((operation_id, inputs))
        self.submitted.set()
        if self.submit_error is not None:
            raise self.submit_error
        return "remote-42"

    async def query(self, ticket):
        self.queries += 1
        if self.query_effects:
            effect = self.query_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            return effect
        return OperationUpdate(
            ticket, self.status, output={"done": True}, feedback={"progress": 1}
        )

    async def cancel(self, ticket):
        self.cancels += 1
        self.cancelled.set()
        if self.cancel_error is not None:
            raise self.cancel_error


def test_remote_completion_uses_protocol_key_and_durable_ticket_checkpoint(tmp_path):
    async def scenario():
        backend = FakeBackend()
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("remote", TicketAction(backend), (ResourceClaim("tool"),))],
                [Resource("tool")],
                journal=journal,
            ) as engine:
                run = engine.submit(
                    Workflow((Node("remote", "remote", {"action": "grip"}),))
                )
                result = await run.wait(1)
                assert result.status == RunStatus.SUCCEEDED
                operation = run.run_id + "/remote"
                assert backend.submissions == [(operation, {"action": "grip"})]
                expected = {"ticket": "remote-42", "operation_id": operation}
                assert result.nodes["remote"].checkpoint == expected
                assert journal.get_run(run.run_id)["nodes"][0]["checkpoint"] == expected
                assert [event["kind"] for event in journal.events(run.run_id)] == [
                    "admitted",
                    "dispatch_intent",
                    "checkpoint",
                    "settled",
                    "finished",
                ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "captured_budget", [None, "poll_interval", "rpc_timeout", "stop_timeout"]
)
def test_cancel_acceptance_does_not_release_resource_before_final_stop(captured_budget):
    class Budget(float):
        def __radd__(self, other):
            raise RuntimeError("custom remote budget arithmetic")

    async def scenario():
        backend = FakeBackend(OperationStatus.RUNNING)
        options = {"poll_interval": 0.001, "rpc_timeout": 1, "stop_timeout": 1}
        if captured_budget is not None:
            options[captured_budget] = Budget(options[captured_budget])
        adapter = TicketAction(backend, **options)
        async with WorkflowEngine(
            [Action("remote", adapter, (ResourceClaim("tool"),))], [Resource("tool")]
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            await asyncio.wait_for(backend.submitted.wait(), 1)
            run.cancel()
            await asyncio.wait_for(backend.cancelled.wait(), 1)
            try:
                with pytest.raises(asyncio.TimeoutError):
                    await run.wait(0.01)
                assert engine.resource_snapshot()["resources"]["tool"]["used"] == 1
                assert run.snapshot().nodes["remote"].status == NodeStatus.CANCELLING
            finally:
                backend.status = OperationStatus.CANCELLED
            assert (await run.wait(1)).status == RunStatus.CANCELLED
            assert backend.cancels == 1
            assert engine.resource_snapshot()["resources"]["tool"]["used"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "submit_error", [ConnectionError("lost response"), TimeoutError("timeout")]
)
def test_lost_submission_is_unknown_and_is_never_retried(submit_error):
    async def scenario():
        backend = FakeBackend()
        backend.submit_error = submit_error
        async with WorkflowEngine(
            [Action("remote", TicketAction(backend), (ResourceClaim("arm"),))],
            [Resource("arm")],
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert len(backend.submissions) == 1
            assert backend.queries == 0
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            engine.confirm_quiescent(
                run.run_id + "/remote", evidence="queried by stable operation key"
            )

    asyncio.run(scenario())


def test_definite_rejection_is_failed_without_quarantine():
    async def scenario():
        backend = FakeBackend()
        backend.submit_error = OperationRejected("interlock refused command")
        async with WorkflowEngine([Action("remote", TicketAction(backend))]) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            result = await run.wait(1)
            assert result.status == RunStatus.FAILED
            assert "OperationRejected" in result.nodes["remote"].error
            assert not engine.resource_snapshot()["quarantined"]

    asyncio.run(scenario())


def test_explicit_backend_uncertainty_is_returned_without_query_retry():
    async def scenario():
        backend = FakeBackend()
        backend.query_effects = [ExecutionUncertain("controller state is unknown")]
        async with WorkflowEngine([Action("remote", TicketAction(backend))]) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            result = await run.wait(1)
            assert result.status == RunStatus.NEEDS_ATTENTION
            assert backend.queries == 1
            assert result.nodes["remote"].error == (
                "ExecutionUncertain: controller state is unknown"
            )
            engine.confirm_quiescent(run.operation_id("remote"), evidence="controller idle")

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["submit", "query", "wait_cancelled"])
@pytest.mark.parametrize("interruption", [False, True], ids=["factory-error", "factory-cancel"])
def test_rpc_task_start_failure_closes_coroutines_and_drains_helpers(stage, interruption):
    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        rejected, queries = [], []
        backend = FakeBackend(OperationStatus.RUNNING)

        async def query(ticket):
            task = asyncio.current_task()
            queries.append(task)
            try:
                await asyncio.Event().wait()
            finally:
                queries.remove(task)

        backend.query = query

        def factory(loop, coroutine, **kwargs):
            if coroutine.cr_code.co_name == stage:
                rejected.append(coroutine)
                if interruption:
                    raise asyncio.CancelledError("RPC task startup interrupted")
                raise RuntimeError("RPC task startup failed")
            return asyncio.Task(coroutine, loop=loop, **kwargs)

        engine = WorkflowEngine(
            [Action("remote", TicketAction(backend, rpc_timeout=0.01, max_query_errors=1),
                    (ResourceClaim("arm"),))],
            [Resource("arm")],
        )
        loop.set_task_factory(factory)
        try:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            result = await run.wait(1)
            assert rejected
            assert all(coroutine.cr_frame is None for coroutine in rejected)
            assert not queries
            expected = RunStatus.FAILED if stage == "submit" else RunStatus.NEEDS_ATTENTION
            assert result.status == expected
            assert len(backend.submissions) == (0 if stage == "submit" else 1)
            assert not any(not task.done() for task in queries)
            if stage != "submit":
                engine.confirm_quiescent(run.operation_id("remote"), evidence="controller idle")
        finally:
            loop.set_task_factory(original_factory)
            for coroutine in rejected:
                coroutine.close()
            for task in list(queries):
                task.cancel()
            await asyncio.gather(*list(queries), return_exceptions=True)
            await engine.aclose(1)

    asyncio.run(scenario())


def test_late_submit_acknowledgement_during_timeout_cleanup_keeps_its_ticket():
    async def scenario():
        class Backend(FakeBackend):
            async def submit(self, operation_id, inputs):
                self.submissions.append((operation_id, inputs))
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return "accepted-during-cleanup"

            async def query(self, ticket):
                return OperationUpdate(ticket, OperationStatus.SUCCEEDED, {"done": True})

        backend = Backend()
        async with WorkflowEngine([
            Action("remote", TicketAction(backend, rpc_timeout=0.005))
        ]) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            result = await run.wait(1)
            assert result.status == RunStatus.SUCCEEDED
            assert result.nodes["remote"].checkpoint["ticket"] == "accepted-during-cleanup"
            assert len(backend.submissions) == 1

    asyncio.run(scenario())


@pytest.mark.skipif(not hasattr(asyncio, "eager_task_factory"), reason="requires eager task support")
def test_factory_failure_after_eager_submit_preserves_unknown_device_lease():
    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        backend = FakeBackend()

        def factory(loop, coroutine, **kwargs):
            if coroutine.cr_code.co_name == "submit":
                asyncio.eager_task_factory(loop, coroutine, **kwargs)
                raise RuntimeError("factory failed after submit ran")
            return asyncio.Task(coroutine, loop=loop, **kwargs)

        async with WorkflowEngine(
            [Action("remote", TicketAction(backend), (ResourceClaim("arm"),))],
            [Resource("arm")],
        ) as engine:
            loop.set_task_factory(factory)
            try:
                run = engine.submit(Workflow((Node("remote", "remote"),)))
                result = await run.wait(1)
                assert result.status == RunStatus.NEEDS_ATTENTION
                assert len(backend.submissions) == 1
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
                engine.confirm_quiescent(run.operation_id("remote"), evidence="controller idle")
            finally:
                loop.set_task_factory(original_factory)

    asyncio.run(scenario())


@pytest.mark.parametrize("eager", [False, True])
def test_factory_failure_after_creating_submit_task_drains_its_owner(eager):
    if eager and not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("requires eager task support")

    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        created, unhandled = [], []
        old_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, detail: unhandled.append(detail))
        closed = asyncio.Event()

        class Backend(FakeBackend):
            async def submit(self, operation_id, inputs):
                self.submissions.append((operation_id, inputs))
                try:
                    await asyncio.Event().wait()
                finally:
                    closed.set()

        backend = Backend()

        def factory(loop, coroutine, **kwargs):
            if coroutine.cr_code.co_name == "submit":
                task = (asyncio.eager_task_factory(loop, coroutine, **kwargs) if eager
                        else asyncio.Task(coroutine, loop=loop, **kwargs))
                created.append(task)
                raise RuntimeError("factory failed after creating its task")
            return asyncio.Task(coroutine, loop=loop, **kwargs)

        async with WorkflowEngine([
            Action("remote", TicketAction(backend), (ResourceClaim("arm"),))
        ], [Resource("arm")]) as engine:
            loop.set_task_factory(factory)
            try:
                run = engine.submit(Workflow((Node("remote", "remote"),)))
                result = await run.wait(1)
                assert result.status == (RunStatus.NEEDS_ATTENTION if eager else RunStatus.FAILED)
                assert created and all(task.done() for task in created)
                assert len(backend.submissions) == int(eager)
                assert closed.is_set() is eager
                if eager:
                    engine.confirm_quiescent(run.operation_id("remote"), evidence="controller idle")
                await asyncio.sleep(0)
                assert not unhandled
            finally:
                loop.set_task_factory(original_factory)
                for task in created:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*created, return_exceptions=True)
                loop.set_exception_handler(old_handler)

    asyncio.run(scenario())


def test_rejected_cancel_rpc_is_closed_before_retrying_stop():
    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        rejected = []

        class Backend(FakeBackend):
            async def cancel(self, ticket):
                await super().cancel(ticket)
                self.status = OperationStatus.CANCELLED

        def factory(loop, coroutine, **kwargs):
            if coroutine.cr_code.co_name == "cancel" and not rejected:
                rejected.append(coroutine)
                raise RuntimeError("could not start cancellation RPC")
            return asyncio.Task(coroutine, loop=loop, **kwargs)

        backend = Backend(OperationStatus.RUNNING)
        async with WorkflowEngine([
            Action("remote", TicketAction(backend, poll_interval=0.001))
        ]) as engine:
            loop.set_task_factory(factory)
            try:
                run = engine.submit(Workflow((Node("remote", "remote"),)))
                await asyncio.wait_for(backend.submitted.wait(), 1)
                run.cancel()
                result = await run.wait(1)
                assert result.status == RunStatus.CANCELLED
                assert backend.cancels == 1
                assert rejected[0].cr_frame is None
            finally:
                loop.set_task_factory(original_factory)
                for coroutine in rejected:
                    coroutine.close()

    asyncio.run(scenario())


def test_repeated_worker_cancellation_cannot_detach_remote_query_cleanup():
    async def scenario():
        entered, cleaning, release, drained = (asyncio.Event() for _ in range(4))

        class Backend(FakeBackend):
            async def query(self, ticket):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaning.set()
                    await release.wait()
                    drained.set()

        engine = WorkflowEngine(
            [Action("remote", TicketAction(Backend()), (ResourceClaim("arm"),))],
            [Resource("arm")], stop_timeout=0.005,
        )
        try:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            await asyncio.wait_for(entered.wait(), 1)
            worker = next(iter(engine._workers))
            worker.cancel()
            await asyncio.wait_for(cleaning.wait(), 1)
            worker.cancel()
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert engine.scheduler_snapshot()["draining_adapters"] == 1
            assert not drained.is_set()
            with pytest.raises(RuntimeError, match="still draining"):
                engine.confirm_quiescent(run.operation_id("remote"), evidence="controller idle")
        finally:
            release.set()
            await engine.aclose(1)
        assert drained.is_set()
        assert engine.scheduler_snapshot()["draining_adapters"] == 0
        assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
        engine.confirm_quiescent(run.operation_id("remote"), evidence="query cleanup completed")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,expected",
    [
        (OperationStatus.FAILED, RunStatus.FAILED),
        (OperationStatus.CANCELLED, RunStatus.CANCELLED),
        (OperationStatus.UNKNOWN, RunStatus.NEEDS_ATTENTION),
    ],
)
def test_final_remote_outcomes_map_to_execution_and_resource_states(status, expected):
    async def scenario():
        async with WorkflowEngine(
            [
                Action(
                    "remote", TicketAction(FakeBackend(status)), (ResourceClaim("arm"),)
                )
            ],
            [Resource("arm")],
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            assert (await run.wait(1)).status == expected
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == int(
                status == OperationStatus.UNKNOWN
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("captured_limit", [False, True])
def test_query_retries_are_bounded_and_reset_after_successful_observations(
    captured_limit,
):
    class Limit(int):
        def __le__(self, other):
            raise RuntimeError("custom query-error comparison")

        def __int__(self):
            raise RuntimeError("custom query-error conversion")

    async def scenario():
        limit = Limit(2) if captured_limit else 2
        backend = FakeBackend()
        backend.query_effects = [
            ConnectionError("transient"),
            OperationUpdate("remote-42", OperationStatus.RUNNING),
            ConnectionError("transient"),
            OperationUpdate("remote-42", OperationStatus.SUCCEEDED),
        ]
        async with WorkflowEngine(
            [
                Action(
                    "remote",
                    TicketAction(backend, poll_interval=0.001, max_query_errors=limit),
                )
            ]
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED
            assert backend.queries == 4
        backend = FakeBackend()
        backend.query_effects = [ConnectionError("offline")] * 3
        async with WorkflowEngine(
            [
                Action(
                    "remote",
                    TicketAction(backend, poll_interval=0.001, max_query_errors=limit),
                    (ResourceClaim("tool"),),
                )
            ],
            [Resource("tool")],
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert backend.queries == 2
            assert len(backend.submissions) == 1
            assert engine.resource_snapshot()["resources"]["tool"]["used"] == 1
            engine.confirm_quiescent(
                run.operation_id("remote"), evidence="tool stopped"
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "cancel_error", [None, ConnectionError("cancel acknowledgement lost")]
)
def test_remote_stop_timeout_is_unknown_even_if_cancel_was_accepted(cancel_error):
    async def scenario():
        backend = FakeBackend(OperationStatus.RUNNING)
        backend.cancel_error = cancel_error
        async with WorkflowEngine(
            [
                Action(
                    "remote",
                    TicketAction(backend, poll_interval=0.001, stop_timeout=0.01),
                )
            ]
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            await asyncio.wait_for(backend.submitted.wait(), 1)
            run.cancel()
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert backend.cancels >= 1
            assert backend.cancels == 1 if cancel_error is None else backend.cancels > 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "update",
    [OperationUpdate("wrong", OperationStatus.SUCCEEDED), {"status": "succeeded"}],
)
def test_invalid_remote_ticket_cannot_finish_another_operation(update):
    async def scenario():
        backend = FakeBackend()
        backend.query_effects = [update]
        async with WorkflowEngine([Action("remote", TicketAction(backend))]) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert backend.queries == 1

    asyncio.run(scenario())


def test_checkpoint_failure_after_acceptance_is_unknown_and_keeps_lease(tmp_path):
    async def scenario():
        backend = FakeBackend()
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [Action("remote", TicketAction(backend), (ResourceClaim("tool"),))],
                [Resource("tool")],
                journal=journal,
            ) as engine:
                with patch.object(
                    journal, "_checkpoint", side_effect=JournalError("disk full")
                ):
                    run = engine.submit(Workflow((Node("remote", "remote"),)))
                    assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
                assert len(backend.submissions) == 1
                assert backend.queries == 0
                assert engine.resource_snapshot()["resources"]["tool"]["used"] == 1

    asyncio.run(scenario())


def test_scoped_tickets_keep_hierarchical_ids_and_persist_all_step_checkpoints(
    tmp_path,
):
    async def scenario():
        backend = FakeBackend()
        scope = workflow_action(
            "scope",
            Workflow((Node("a", "remote"), Node("b", "remote", after=("a",)))),
            [Action("remote", TicketAction(backend))],
            [Resource("tool")],
        )
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [scope], [Resource("tool")], journal=journal
            ) as engine:
                run = engine.submit(Workflow((Node("scope", "scope"),)))
                result = await run.wait(1)
                assert result.status == RunStatus.SUCCEEDED
                prefix = run.run_id + "/scope/"
                assert [operation for operation, _ in backend.submissions] == [
                    prefix + "a",
                    prefix + "b",
                ]
                checkpoints = journal.get_run(run.run_id)["nodes"][0]["checkpoint"][
                    "scope_nodes"
                ]
                assert checkpoints == {
                    "a": {"ticket": "remote-42", "operation_id": prefix + "a"},
                    "b": {"ticket": "remote-42", "operation_id": prefix + "b"},
                }

    asyncio.run(scenario())


def test_oversized_optional_remote_progress_cannot_release_active_resources():
    async def scenario():
        backend = FakeBackend()
        backend.query_effects = [
            OperationUpdate(
                "remote-42", OperationStatus.RUNNING, feedback={"huge": "x" * 1000}
            ),
            OperationUpdate(
                "remote-42", OperationStatus.SUCCEEDED, output={"done": True}
            ),
        ]
        async with WorkflowEngine(
            [Action("remote", TicketAction(backend, poll_interval=0.001))],
            max_feedback_bytes_per_report=32,
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED
            assert backend.queries == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "option,value",
    [
        ("rpc_timeout", 0),
        ("poll_interval", None),
        ("stop_timeout", True),
        ("max_query_errors", 0),
    ],
)
def test_remote_adapter_rejects_invalid_limits_and_synchronous_transports(
    option, value
):
    async def scenario():
        with pytest.raises(ValueError):
            TicketAction(FakeBackend(), **{option: value})

    asyncio.run(scenario())
    with pytest.raises(TypeError, match="must be async"):
        TicketAction(object())


def test_late_submit_ticket_is_durable_after_engine_stop_deadline(tmp_path):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class DelayedBackend(FakeBackend):
            async def submit(self, operation_id, inputs):
                entered.set()
                await release.wait()
                return await super().submit(operation_id, inputs)

            async def cancel(self, ticket):
                await super().cancel(ticket)
                self.status = OperationStatus.CANCELLED

        backend = DelayedBackend(OperationStatus.RUNNING)
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [
                    Action(
                        "remote",
                        TicketAction(backend, poll_interval=0.001),
                        (ResourceClaim("tool"),),
                    )
                ],
                [Resource("tool")],
                journal=journal,
                stop_timeout=0.01,
            ) as engine:
                run = engine.submit(Workflow((Node("remote", "remote"),)))
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                    run.cancel()
                    assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
                    assert run.snapshot().nodes["remote"].checkpoint is None
                finally:
                    release.set()
                await engine.aclose(1)
                record = journal.get_run(run.run_id)
                assert record["nodes"][0]["checkpoint"]["ticket"] == "remote-42"
                assert backend.cancels == 1
                assert record["nodes"][0]["status"] == "unknown"
                assert record["nodes"][0]["lease_active"]
                engine.confirm_quiescent(
                    run.run_id + "/remote", evidence="late ticket queried and cancelled"
                )

    asyncio.run(scenario())


def test_blocked_remote_query_cannot_report_stop_confirmation_after_its_deadline():
    import time

    async def scenario():
        class Backend(FakeBackend):
            async def query(self, ticket):
                if self.cancelled.is_set():
                    time.sleep(0.02)
                    return OperationUpdate(ticket, OperationStatus.CANCELLED)
                return await super().query(ticket)

        backend = Backend(OperationStatus.RUNNING)
        async with WorkflowEngine(
            [
                Action(
                    "remote",
                    TicketAction(backend, stop_timeout=0.005, poll_interval=0.001),
                )
            ]
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            await asyncio.wait_for(backend.submitted.wait(), 1)
            run.cancel()
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION

    asyncio.run(scenario())


def test_stop_interrupts_slow_query_and_drains_it_before_sending_cancel():
    async def scenario():
        entered, drained = asyncio.Event(), asyncio.Event()

        class Backend(FakeBackend):
            async def query(self, ticket):
                if self.cancelled.is_set():
                    return OperationUpdate(ticket, OperationStatus.CANCELLED)
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()

            async def cancel(self, ticket):
                assert drained.is_set()
                await super().cancel(ticket)

        backend = Backend(OperationStatus.RUNNING)
        async with WorkflowEngine(
            [Action("remote", TicketAction(backend, rpc_timeout=30))],
            stop_timeout=0.5,
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            await asyncio.wait_for(entered.wait(), 1)
            run.cancel()
            assert (await run.wait(1)).status == RunStatus.CANCELLED
            assert backend.cancels == 1
            assert not engine.resource_snapshot()["quarantined"]

    asyncio.run(scenario())


def test_interrupted_adapter_drains_query_helpers_and_keeps_unknown_lease():
    async def scenario():
        entered, drained = asyncio.Event(), asyncio.Event()

        class Backend(FakeBackend):
            async def query(self, ticket):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()

        async with WorkflowEngine(
            [Action("remote", TicketAction(Backend()), (ResourceClaim("arm"),))],
            [Resource("arm")],
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            await asyncio.wait_for(entered.wait(), 1)
            for worker in tuple(engine._workers):
                worker.cancel()
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert drained.is_set()
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            await engine.aclose(1)
            assert not engine._workers

    asyncio.run(scenario())


def test_none_query_response_is_invalid_rather_than_a_stop_notification():
    async def scenario():
        backend = FakeBackend()
        backend.query_effects = [None]
        async with WorkflowEngine([Action("remote", TicketAction(backend))]) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert backend.queries == 1

    asyncio.run(scenario())


def test_query_cleanup_time_counts_toward_remote_stop_confirmation_budget():
    async def scenario():
        entered, cleaned = asyncio.Event(), asyncio.Event()

        class Backend(FakeBackend):
            async def query(self, ticket):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0.03)
                    cleaned.set()

        backend = Backend()
        async with WorkflowEngine(
            [Action("remote", TicketAction(backend, stop_timeout=0.005))]
        ) as engine:
            run = engine.submit(Workflow((Node("remote", "remote"),)))
            await asyncio.wait_for(entered.wait(), 1)
            run.cancel()
            assert (await run.wait(1)).status == RunStatus.NEEDS_ATTENTION
            assert cleaned.is_set()
            assert "deadline expired" in run.snapshot().nodes["remote"].error

    asyncio.run(scenario())
