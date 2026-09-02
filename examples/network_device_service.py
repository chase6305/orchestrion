"""Small HTTP device service used by the network orchestration demo."""

import json
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused for a different command."""


class SimulatedNetworkDevice:
    """Execute ticketed commands and expose completion through long polling."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_operation_id = 0
        self._operations: Dict[str, Dict] = {}
        self._idempotency_keys: Dict[str, str] = {}
        self._cancel_events: Dict[str, threading.Event] = {}

    def submit(self, command: Dict, idempotency_key: str) -> Dict:
        with self._condition:
            existing = self._idempotency_keys.get(idempotency_key)
            if existing is not None:
                if self._operations[existing]["command"] != command:
                    raise IdempotencyConflictError(
                        "idempotency key belongs to a different command"
                    )
                return {"operation_id": existing, "deduplicated": True}
            operation_id = str(self._next_operation_id)
            self._next_operation_id += 1
            self._idempotency_keys[idempotency_key] = operation_id
            self._operations[operation_id] = {
                "operation_id": operation_id,
                "status": "running",
                "command": command.copy(),
            }
            self._cancel_events[operation_id] = threading.Event()
        worker = threading.Thread(
            target=self._execute,
            args=(operation_id,),
            name="simulated-network-device-command",
            daemon=True,
        )
        worker.start()
        return {"operation_id": operation_id, "deduplicated": False}

    def _execute(self, operation_id: str) -> None:
        with self._condition:
            command = self._operations[operation_id]["command"]
            cancel_event = self._cancel_events[operation_id]
        if cancel_event.wait(command.get("duration", 0.02)):
            return
        with self._condition:
            if self._operations[operation_id]["status"] != "running":
                return
            self._operations[operation_id].update(
                {
                    "status": "succeeded",
                    "result": {
                        "action": command["action"],
                        "accepted": True,
                    },
                    "completed_at": time.time(),
                }
            )
            self._condition.notify_all()

    def cancel(self, operation_id: str) -> Optional[Dict]:
        with self._condition:
            operation = self._operations.get(operation_id)
            if operation is None:
                return None
            cancelled = operation["status"] == "running"
            if cancelled:
                operation["status"] = "cancelled"
                operation["error"] = "cancelled by client"
                self._cancel_events[operation_id].set()
                self._condition.notify_all()
            return {
                "operation_id": operation_id,
                "status": operation["status"],
                "cancelled": cancelled,
            }

    def cancel_all(self, reason: str = "device server stopped") -> List[str]:
        """Cancel every running operation and wake all response waiters."""
        with self._condition:
            operation_ids = [
                operation_id
                for operation_id, operation in self._operations.items()
                if operation["status"] == "running"
            ]
            for operation_id in operation_ids:
                self._operations[operation_id]["status"] = "cancelled"
                self._operations[operation_id]["error"] = reason
                self._cancel_events[operation_id].set()
            if operation_ids:
                self._condition.notify_all()
            return operation_ids

    def wait(self, operation_id: str, timeout: float) -> Optional[Dict]:
        deadline = time.monotonic() + timeout
        with self._condition:
            operation = self._operations.get(operation_id)
            if operation is None:
                return None
            while operation["status"] == "running":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return operation.copy()

    def health(self) -> Dict:
        with self._condition:
            running = sum(
                operation["status"] == "running"
                for operation in self._operations.values()
            )
            cancelled = sum(
                operation["status"] == "cancelled"
                for operation in self._operations.values()
            )
        return {
            "health": "online",
            "available": True,
            "observed_at": time.time(),
            "running_operations": running,
            "cancelled_operations": cancelled,
        }


class NetworkDeviceServer:
    """Context-managed HTTP wrapper around :class:`SimulatedNetworkDevice`."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.device = SimulatedNetworkDevice()
        handler = self._make_handler(self.device)
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread: Optional[threading.Thread] = None
        self._lifecycle_lock = threading.Lock()
        self._closed = False

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return "http://{}:{}".format(host, port)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("network device server is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="simulated-network-device-http",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            thread = self._thread
            if thread is not None and thread.is_alive():
                self._server.shutdown()
            self.device.cancel_all()
            self._server.server_close()
            self._closed = True
        if thread is not None:
            thread.join(timeout=1.0)

    def __enter__(self) -> "NetworkDeviceServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    @staticmethod
    def _make_handler(device: SimulatedNetworkDevice):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._send_json(HTTPStatus.OK, device.health())
                    return
                if parsed.path.startswith("/operations/"):
                    operation_id = parsed.path.rsplit("/", 1)[-1]
                    try:
                        wait = float(parse_qs(parsed.query).get("wait", ["0"])[0])
                    except ValueError:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid wait"})
                        return
                    if not math.isfinite(wait) or wait < 0:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid wait"})
                        return
                    operation = device.wait(operation_id, min(wait, 1.0))
                    if operation is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown operation"})
                    else:
                        status = (
                            HTTPStatus.OK
                            if operation["status"] != "running"
                            else HTTPStatus.ACCEPTED
                        )
                        self._send_json(status, operation)
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:
                if self.path != "/commands":
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                idempotency_key = self.headers.get("Idempotency-Key")
                if not idempotency_key:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST, {"error": "missing Idempotency-Key"}
                    )
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    command = json.loads(self.rfile.read(length))
                    if not self._valid_command(command):
                        raise ValueError
                except (ValueError, json.JSONDecodeError):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid command"})
                    return
                try:
                    accepted = device.submit(command, idempotency_key)
                except IdempotencyConflictError as exc:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
                    return
                self._send_json(HTTPStatus.ACCEPTED, accepted)

            def do_DELETE(self) -> None:
                parsed = urlparse(self.path)
                if not parsed.path.startswith("/operations/"):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                operation_id = parsed.path.rsplit("/", 1)[-1]
                result = device.cancel(operation_id)
                if result is None:
                    self._send_json(
                        HTTPStatus.NOT_FOUND, {"error": "unknown operation"}
                    )
                    return
                self._send_json(HTTPStatus.OK, result)

            @staticmethod
            def _valid_command(command: object) -> bool:
                if not isinstance(command, dict):
                    return False
                if set(command) != {"action", "duration"}:
                    return False
                duration = command["duration"]
                return (
                    command["action"] in {"grip", "release", "inspect"}
                    and not isinstance(duration, bool)
                    and isinstance(duration, (int, float))
                    and math.isfinite(duration)
                    and 0 <= duration <= 2.0
                )

            def _send_json(self, status: HTTPStatus, payload: Dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args) -> None:
                return

        return Handler


if __name__ == "__main__":
    with NetworkDeviceServer(port=8080) as server:
        print("Network device listening at {} (Ctrl-C to stop)".format(server.base_url))
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
