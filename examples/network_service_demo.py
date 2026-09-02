"""Coordinate a ticket-based HTTP device and wait for its remote response."""

import concurrent.futures
import json
import math
import time
from numbers import Real
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from examples.network_device_service import NetworkDeviceServer
from orchestrion import (
    CallableTask,
    FieldSpec,
    MoveSyncOption,
    RequestSchema,
    RetryPolicy,
    SimulatedRobotTask,
    TaskPilot,
)


class NetworkDeviceClient:
    """Synchronous adapter for a ticket-based asynchronous HTTP protocol."""

    def __init__(self, base_url: str, request_timeout: float = 0.5) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if (
            isinstance(request_timeout, bool)
            or not isinstance(request_timeout, Real)
            or not math.isfinite(request_timeout)
            or request_timeout <= 0
        ):
            raise ValueError("request_timeout must be positive and finite")
        self.base_url = base_url.rstrip("/")
        self.request_timeout = float(request_timeout)

    def execute(self, request_id: int, content: Dict) -> Dict:
        operation = self._request_json(
            "/commands",
            method="POST",
            payload={"action": content["action"], "duration": content["duration"]},
            headers={"Idempotency-Key": content["operation_key"]},
        )
        operation_id = operation["operation_id"]
        deadline = time.monotonic() + content["response_timeout"]
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    self.cancel(operation_id)
                except (ConnectionError, RuntimeError):
                    # Preserve the primary timeout; reconciliation can use the
                    # stable operation key after connectivity is restored.
                    pass
                raise TimeoutError(
                    "remote operation {} did not finish in time".format(operation_id)
                )
            result = self._request_json(
                "/operations/{}?{}".format(
                    operation_id, urlencode({"wait": min(remaining, 0.2)})
                )
            )
            if result["status"] != "running":
                if result["status"] != "succeeded":
                    raise RuntimeError(
                        "remote operation {} ended as {}: {}".format(
                            operation_id,
                            result["status"],
                            result.get("error", "no error detail"),
                        )
                    )
                return {
                    "request_id": request_id,
                    "operation_id": operation_id,
                    "deduplicated": operation["deduplicated"],
                    "remote_completed_at": result["completed_at"],
                    **result["result"],
                }

    def cancel(self, operation_id: str) -> Dict:
        return self._request_json(
            "/operations/{}".format(operation_id), method="DELETE"
        )

    def status(self) -> Dict:
        return self._request_json("/health")

    def _request_json(
        self,
        path: str,
        method: str = "GET",
        payload: Dict = None,
        headers: Dict = None,
    ) -> Dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("device returned HTTP {}: {}".format(exc.code, detail))
        except (URLError, OSError) as exc:
            raise ConnectionError("device request failed: {}".format(exc)) from exc


def run_workflow(base_url: str) -> Dict:
    """Run one move-synchronized remote command and return its final result."""
    client = NetworkDeviceClient(base_url)
    remote_tool = CallableTask(
        client.execute,
        status=client.status,
        retry_policy=RetryPolicy(
            max_attempts=3,
            delay=0.02,
            backoff=2.0,
            retry_exceptions=(ConnectionError,),
        ),
        request_schema=RequestSchema(
            {
                "action": FieldSpec(
                    "string", required=True, choices=("grip", "release")
                ),
                "duration": FieldSpec(
                    "number", required=True, minimum=0.0, maximum=2.0
                ),
                "response_timeout": FieldSpec(
                    "number", required=True, minimum=0.01, maximum=10.0
                ),
                "operation_key": FieldSpec("string", required=True),
            }
        ),
        metadata={
            "transport": "http",
            "protocol": "ticket-and-long-poll",
            "endpoints": [
                "POST /commands",
                "GET /operations/{id}",
                "DELETE /operations/{id}",
            ],
        },
    )
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    with TaskPilot(
        SimulatedRobotTask([0.0, 0.0]),
        {"remote_tool": remote_tool},
        executor_owned=executor,
    ) as pilot:
        _, move_end = pilot.move_joint_trajectory_async(
            [[0.2, 0.1], [0.7, 0.4]], interval=0.01
        )
        request_id = pilot.call_srv_async(
            "remote_tool",
            {
                "action": "grip",
                "duration": 0.02,
                "response_timeout": 1.0,
                "operation_key": "demo-cycle-1/grip",
            },
            MoveSyncOption.sync_w_explicit_id(move_end - 1),
            idempotency_key="demo-cycle-1/grip",
        )
        result = pilot.wait_request(request_id, timeout=2.0)
        return {
            "result": result.content,
            "timeline": [event.status.value for event in pilot.timeline(request_id)],
            "service": pilot.describe_services()["remote_tool"],
            "health": pilot.query_task_status("remote_tool"),
        }


def main() -> None:
    with NetworkDeviceServer() as server:
        summary = run_workflow(server.base_url)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
