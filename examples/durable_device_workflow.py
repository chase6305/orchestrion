"""Run a resource-scoped HTTP workflow and replay its durable result after restart.

Run ``python -m examples.durable_device_workflow`` for an ephemeral database, or
pass ``--journal /path/to/cell.sqlite3`` to retain the result. Reusing the same
cycle key replays the previous outcome and does not move the device again.
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote

from examples.network_device_service import NetworkDeviceServer
from examples.network_service_demo import NetworkDeviceClient
from orchestrion import FieldSpec, RequestSchema
from orchestrion.workflow import (
    Action,
    Node,
    OperationStatus,
    OperationUpdate,
    Resource,
    ResourceClaim,
    SQLiteJournal,
    TicketAction,
    Workflow,
    WorkflowEngine,
    workflow_action,
)


class HttpOperationBackend:
    """Adapt the demo's bounded HTTP client to the async ticket protocol.

    Blocking urllib calls run in the event loop's thread pool and have their own
    socket timeout. Cancelling a coroutine cannot interrupt a urllib worker; any
    ambiguous submit remains UNKNOWN and is reconciled using its operation key.
    """

    def __init__(self, base_url: str) -> None:
        self._client = NetworkDeviceClient(base_url, request_timeout=0.2)
        self.submissions = 0

    async def submit(self, operation_id: str, inputs: Dict[str, Any]) -> str:
        self.submissions += 1
        result = await asyncio.to_thread(
            self._client._request_json,
            "/commands",
            method="POST",
            payload=inputs,
            headers={"Idempotency-Key": operation_id},
        )
        return result["operation_id"]

    async def query(self, ticket: str) -> OperationUpdate:
        result = await asyncio.to_thread(
            self._client._request_json,
            "/operations/{}?wait=0.02".format(quote(ticket, safe="")),
        )
        return OperationUpdate(
            result["operation_id"],
            OperationStatus(result["status"]),
            output=result.get("result"),
            error=result.get("error"),
        )

    async def cancel(self, ticket: str) -> None:
        await asyncio.to_thread(self._client.cancel, ticket)


def transfer_action(backend: HttpOperationBackend) -> Action:
    """Keep tool and work-zone ownership across grip and release steps."""
    tool = Action(
        "tool.command",
        TicketAction(backend, poll_interval=0.005, rpc_timeout=0.5),
        (ResourceClaim("tool"),),
        input_schema=RequestSchema(
            {
                "action": FieldSpec(
                    "string", required=True, choices=["grip", "release"]
                ),
                "duration": FieldSpec("number", required=True, minimum=0, maximum=2),
            }
        ),
    )
    return workflow_action(
        "cell.transfer",
        Workflow(
            (
                Node("grip", "tool.command", {"action": "grip", "duration": 0.02}),
                Node(
                    "release",
                    "tool.command",
                    {"action": "release", "duration": 0.02},
                    after=("grip",),
                ),
            )
        ),
        [tool],
        [Resource("tool"), Resource("work_zone")],
    )


async def run_demo(
    base_url: str, journal_path: Path, cycle_key: str = "cycle-42"
) -> Dict:
    backend = HttpOperationBackend(base_url)
    action = transfer_action(backend)
    workflow = Workflow((Node("transfer", "cell.transfer", stop_timeout=1),), timeout=2)
    resources = [Resource("tool"), Resource("work_zone")]
    with SQLiteJournal(journal_path) as journal:
        async with WorkflowEngine(
            [action],
            resources,
            journal=journal,
            max_input_bytes_per_node=4096,
            max_output_bytes_per_node=65536,
            max_feedback_bytes_per_report=4096,
        ) as engine:
            run = engine.submit(workflow, idempotency_key=cycle_key)
            initial = await run.wait(3)
            run_id = run.run_id
    submissions_before_restart = backend.submissions
    with SQLiteJournal(journal_path) as journal:
        async with WorkflowEngine([action], resources, journal=journal) as engine:
            recovered = engine.submit(workflow, idempotency_key=cycle_key)
            result = await recovered.wait(1)
            assert recovered.run_id == run_id
            assert backend.submissions == submissions_before_restart
            return {
                "run_id": run_id,
                "initial_status": initial.status.value,
                "recovered_status": result.status.value,
                "device_submissions": backend.submissions,
                "replayed_without_device_io": True,
                "checkpoint": result.nodes["transfer"].checkpoint,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--cycle-key", default="cycle-42")
    args = parser.parse_args()
    with NetworkDeviceServer() as server:
        if args.journal is None:
            with tempfile.TemporaryDirectory(
                prefix="orchestrion-journal-"
            ) as directory:
                result = asyncio.run(
                    run_demo(
                        server.base_url,
                        Path(directory) / "cell.sqlite3",
                        args.cycle_key,
                    )
                )
        else:
            result = asyncio.run(
                run_demo(server.base_url, args.journal, args.cycle_key)
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
