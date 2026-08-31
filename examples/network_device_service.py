"""Small HTTP device service used by the network orchestration demo."""

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse


class SimulatedNetworkDevice:
    """Execute ticketed commands and expose completion through long polling."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_operation_id = 0
        self._operations: Dict[str, Dict] = {}
        self._idempotency_keys: Dict[str, str] = {}

    def submit(self, command: Dict, idempotency_key: str) -> Dict:
        with self._condition:
            existing = self._idempotency_keys.get(idempotency_key)
            if existing is not None:
                return {"operation_id": existing, "deduplicated": True}
            operation_id = str(self._next_operation_id)
            self._next_operation_id += 1
            self._idempotency_keys[idempotency_key] = operation_id
            self._operations[operation_id] = {
                "operation_id": operation_id,
                "status": "running",
                "command": command.copy(),
            }
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
        time.sleep(command.get("duration", 0.02))
        with self._condition:
            self._operations[operation_id].update(
                {
                    "status": "succeeded",
                    "result": {
                        "action": command["action"],
                        "accepted": True,
                    },
                }
            )
            self._condition.notify_all()

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
        return {
            "health": "online",
            "available": True,
            "observed_at": time.time(),
            "running_operations": running,
        }


class NetworkDeviceServer:
    """Context-managed HTTP wrapper around :class:`SimulatedNetworkDevice`."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.device = SimulatedNetworkDevice()
        handler = self._make_handler(self.device)
        self._server = ThreadingHTTPServer((host, port), handler)
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return "http://{}:{}".format(host, port)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="simulated-network-device-http",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

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
                    operation = device.wait(operation_id, max(0.0, min(wait, 1.0)))
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
                    if not isinstance(command, dict) or "action" not in command:
                        raise ValueError
                except (ValueError, json.JSONDecodeError):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid command"})
                    return
                self._send_json(
                    HTTPStatus.ACCEPTED,
                    device.submit(command, idempotency_key),
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
