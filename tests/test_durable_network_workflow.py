"""Exercise the durable workflow against a real loopback HTTP device service."""

import asyncio

from examples.durable_device_workflow import HttpOperationBackend, run_demo
from examples.network_device_service import NetworkDeviceServer
from orchestrion.workflow import (
    Action,
    Node,
    OperationStatus,
    Resource,
    ResourceClaim,
    RunStatus,
    SQLiteJournal,
    TicketAction,
    Workflow,
    WorkflowEngine,
)


def test_http_scope_replays_after_engine_and_journal_restart_without_new_commands(
    tmp_path,
):
    with NetworkDeviceServer() as server:
        first = asyncio.run(run_demo(server.base_url, tmp_path / "cell.db"))
        assert first["initial_status"] == first["recovered_status"] == "succeeded"
        assert first["device_submissions"] == 2
        assert first["replayed_without_device_io"]
        checkpoints = first["checkpoint"]["scope_nodes"]
        assert set(checkpoints) == {"grip", "release"}
        second = asyncio.run(run_demo(server.base_url, tmp_path / "cell.db"))
        assert second["run_id"] == first["run_id"]
        assert second["device_submissions"] == 0
        assert server.device._next_operation_id == 2


def test_http_cancellation_waits_for_remote_terminal_result(tmp_path):
    async def scenario(base_url):
        submitted = asyncio.Event()

        class Backend(HttpOperationBackend):
            async def submit(self, operation_id, inputs):
                ticket = await super().submit(operation_id, inputs)
                submitted.set()
                return ticket

        backend = Backend(base_url)
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [
                    Action(
                        "tool",
                        TicketAction(backend, poll_interval=0.001),
                        (ResourceClaim("tool"),),
                    )
                ],
                [Resource("tool")],
                journal=journal,
            ) as engine:
                run = engine.submit(
                    Workflow((Node("grip", "tool", {"action": "grip", "duration": 2}),))
                )
                await asyncio.wait_for(submitted.wait(), 1)
                run.cancel()
                result = await run.wait(2)
                assert result.status == RunStatus.CANCELLED
                ticket = result.nodes["grip"].checkpoint["ticket"]
                assert (await backend.query(ticket)).status == OperationStatus.CANCELLED
                assert engine.resource_snapshot()["resources"]["tool"]["used"] == 0
                assert journal.get_run(run.run_id)["nodes"][0]["lease_active"] is False

    with NetworkDeviceServer() as server:
        asyncio.run(scenario(server.base_url))


def test_lost_http_submit_response_retains_resource_through_restart(tmp_path):
    command = {"action": "grip", "duration": 2}

    async def first(base_url):
        class LostResponse(HttpOperationBackend):
            async def submit(self, operation_id, inputs):
                await super().submit(operation_id, inputs)
                raise ConnectionError("connection lost after device accepted command")

        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [
                    Action(
                        "tool",
                        TicketAction(LostResponse(base_url)),
                        (ResourceClaim("tool"),),
                    )
                ],
                [Resource("tool")],
                journal=journal,
            ) as engine:
                run = engine.submit(
                    Workflow((Node("grip", "tool", command),)), idempotency_key="cycle"
                )
                assert (await run.wait(2)).status == RunStatus.NEEDS_ATTENTION
                assert engine.resource_snapshot()["resources"]["tool"]["used"] == 1
                return run.run_id

    async def recover(base_url, run_id, ticket):
        backend = HttpOperationBackend(base_url)
        with SQLiteJournal(tmp_path / "cell.db") as journal:
            async with WorkflowEngine(
                [], [Resource("tool")], journal=journal
            ) as engine:
                assert engine.get_run_by_key("cycle").run_id == run_id
                assert engine.resource_snapshot()["resources"]["tool"]["used"] == 1
                await backend.cancel(ticket)
                assert (await backend.query(ticket)).status == OperationStatus.CANCELLED
                engine.confirm_quiescent(
                    run_id + "/grip",
                    evidence="remote operation query confirmed cancellation",
                )
                assert engine.resource_snapshot()["resources"]["tool"]["used"] == 0

    with NetworkDeviceServer() as server:
        run_id = asyncio.run(first(server.base_url))
        # This test device's controller log correlates the stable protocol key
        # even though the transport never delivered a ticket to the adapter.
        ticket = server.device._idempotency_keys[run_id + "/grip"]
        assert server.device._next_operation_id == 1
        asyncio.run(recover(server.base_url, run_id, ticket))
        assert server.device._next_operation_id == 1
