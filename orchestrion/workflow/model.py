"""Transport-independent workflow definitions and execution snapshots."""

import math
from dataclasses import dataclass, field
from enum import Enum
from json.encoder import encode_basestring_ascii
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from urllib.parse import quote


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a nonempty string".format(label))


def _operation_id(run_id: str, node_id: str) -> str:
    """Encode one node component so nested paths cannot alias literal slashes."""
    return run_id + "/" + quote(node_id, safe="", errors="surrogatepass")


def _positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer".format(label))


def _timeout(value: Optional[float]) -> Optional[float]:
    """Capture a primitive budget before it reaches event-loop timer arithmetic."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Preserve integral JSON fingerprints used by existing durable keys.
        value = int.__int__(value) if isinstance(value, int) else float.__float__(value)
        if 0 < value <= 1e12 and math.isfinite(value):
            return value
    raise ValueError("timeout must be finite, positive, at most 1e12, or None")


@dataclass(frozen=True)
class Resource:
    """A named capacity shared by every workflow in one engine."""

    name: str
    capacity: int = 1

    def __post_init__(self) -> None:
        _name(self.name, "resource name")
        _positive_int(self.capacity, "resource capacity")
        object.__setattr__(self, "capacity", int.__int__(self.capacity))


@dataclass(frozen=True)
class ResourceClaim:
    resource: str
    units: int = 1

    def __post_init__(self) -> None:
        _name(self.resource, "claimed resource")
        _positive_int(self.units, "claim units")
        object.__setattr__(self, "units", int.__int__(self.units))


@dataclass(frozen=True)
class OutputRef:
    """Bind an input to an upstream JSON output; also implies a dependency."""

    node_id: str
    path: Tuple[Union[str, int], ...] = ()

    def __post_init__(self) -> None:
        _name(self.node_id, "output node ID")
        if isinstance(self.path, (str, bytes)):
            raise TypeError("output path must be a sequence of keys or indices")
        object.__setattr__(self, "path", tuple(self.path))
        for part in self.path:
            if isinstance(part, bool) or not isinstance(part, (str, int)):
                raise TypeError(
                    "output path must contain string keys or integer indices"
                )
            if isinstance(part, int) and part < 0:
                raise ValueError("output indices must be nonnegative")


def _snapshot(
    value: Any, *, references: bool = False, max_bytes: Optional[int] = None
) -> Any:
    """Copy finite JSON without deepcopy hooks; reject cycles and nonfinite data."""
    if max_bytes is None and (value is None or isinstance(value, (str, bool, int))):
        return value
    ancestors = set()
    remaining = max_bytes

    if max_bytes is not None:

        def charge(amount: int) -> None:
            nonlocal remaining
            if remaining is not None:
                remaining -= amount
                if remaining < 0:
                    raise PayloadLimitError(
                        "JSON payload exceeds {} bytes".format(max_bytes)
                    )

        def scalar(item: Any) -> None:
            if isinstance(item, str):
                # Reject huge strings before allocating their escaped representation.
                text_length = str.__len__(item)
                charge(text_length + 2)
                charge(len(encode_basestring_ascii(item)) - text_length - 2)
            elif item is None or item is True:
                charge(4)  # null / true
            elif item is False:
                charge(5)
            elif isinstance(item, int):
                # A lower decimal-digit bound avoids converting enormous integers.
                # 0.30102 is a lower bound for log10(2); bit_length covers both signs.
                minimum_size = max(
                    1, (int.bit_length(item) - 1) * 30102 // 100000 + 1
                )
                if int.__lt__(item, 0):
                    minimum_size += 1
                charge(minimum_size)
                # JSON uses the base numeric repr, bypassing subclass overrides.
                charge(len(int.__repr__(item)) - minimum_size)
            else:
                # visit() only passes finite floats through this branch.
                charge(len(float.__repr__(item)))

    def visit(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            if remaining is not None:
                scalar(item)
            return item
        if isinstance(item, float) and math.isfinite(item):
            if remaining is not None:
                scalar(item)
            return item
        if references and isinstance(item, OutputRef):
            if remaining is not None:
                visit({"$output": {"node_id": item.node_id, "path": list(item.path)}})
            return item
        if isinstance(item, (dict, list, tuple)):
            if id(item) in ancestors:
                raise ValueError("workflow data must not contain cycles")
            ancestors.add(id(item))
            try:
                if remaining is not None:
                    charge(2)
                if isinstance(item, dict):
                    result = {}
                    for key, child in item.items():
                        if not isinstance(key, str):
                            raise TypeError(
                                "workflow data requires string dictionary keys"
                            )
                        if remaining is not None:
                            scalar(key)
                            charge(1)
                        result[key] = visit(child)
                else:
                    result = [visit(child) for child in item]
                if remaining is not None:
                    # Count separators from copied entries, not subclass lengths.
                    charge(max(0, len(result) - 1))
                return result
            finally:
                ancestors.remove(id(item))
        raise TypeError("workflow data must be finite JSON values or input OutputRefs")

    return visit(value)


class FailurePolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    CONTINUE_INDEPENDENT = "continue_independent"


@dataclass(frozen=True)
class Node:
    node_id: str
    action: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    after: Tuple[str, ...] = ()
    claims: Tuple[ResourceClaim, ...] = ()
    priority: int = 0
    timeout: Optional[float] = None
    stop_timeout: Optional[float] = None

    def __post_init__(self) -> None:
        _name(self.node_id, "node ID")
        _name(self.action, "action name")
        if not isinstance(self.inputs, Mapping):
            raise TypeError("node inputs must be a mapping")
        object.__setattr__(
            self, "inputs", _snapshot(dict(self.inputs), references=True)
        )
        if isinstance(self.after, (str, bytes)):
            raise TypeError("after must be a sequence of node IDs")
        object.__setattr__(self, "after", tuple(self.after))
        for dependency in self.after:
            _name(dependency, "dependency")
        object.__setattr__(self, "claims", tuple(self.claims))
        if any(not isinstance(claim, ResourceClaim) for claim in self.claims):
            raise TypeError("claims must contain ResourceClaim objects")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")
        object.__setattr__(self, "priority", int.__int__(self.priority))
        if self.timeout is not None:
            object.__setattr__(self, "timeout", _timeout(self.timeout))
        if self.stop_timeout is not None:
            object.__setattr__(self, "stop_timeout", _timeout(self.stop_timeout))


@dataclass(frozen=True)
class Workflow:
    nodes: Tuple[Node, ...]
    failure_policy: FailurePolicy = FailurePolicy.FAIL_FAST
    timeout: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        if not self.nodes or any(not isinstance(node, Node) for node in self.nodes):
            raise ValueError("workflow requires at least one Node")
        object.__setattr__(self, "failure_policy", FailurePolicy(self.failure_policy))
        if self.timeout is not None:
            object.__setattr__(self, "timeout", _timeout(self.timeout))


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = frozenset(
    {
        NodeStatus.SUCCEEDED,
        NodeStatus.FAILED,
        NodeStatus.CANCELLED,
        NodeStatus.TIMED_OUT,
        NodeStatus.SKIPPED,
        NodeStatus.UNKNOWN,
    }
)


class RunStatus(str, Enum):
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_ATTENTION = "needs_attention"


@dataclass(frozen=True)
class ActionOutcome:
    """An adapter's explicit result, including failures returned without raising.

    FAILED and CANCELLED confirm device quiescence. UNKNOWN keeps resource leases
    quarantined. Deadline outcomes belong to the engine, not the adapter.
    """

    status: NodeStatus
    output: Any = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        status = NodeStatus(self.status)
        if status not in (
            NodeStatus.SUCCEEDED,
            NodeStatus.FAILED,
            NodeStatus.CANCELLED,
            NodeStatus.UNKNOWN,
        ):
            raise ValueError(
                "action outcome must be succeeded, failed, cancelled or unknown"
            )
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("action error must be a string or None")
        object.__setattr__(self, "status", status)

    @classmethod
    def failed(cls, error: str) -> "ActionOutcome":
        return cls(NodeStatus.FAILED, error=error)

    @classmethod
    def cancelled(cls, error: Optional[str] = None) -> "ActionOutcome":
        return cls(NodeStatus.CANCELLED, error=error)

    @classmethod
    def unknown(cls, error: str) -> "ActionOutcome":
        return cls(NodeStatus.UNKNOWN, error=error)


@dataclass(frozen=True)
class NodeResult:
    node_id: str
    action: str
    status: NodeStatus = NodeStatus.PENDING
    output: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    feedback_dropped: int = 0
    checkpoint: Any = None
    action_version: Optional[str] = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    nodes: Dict[str, NodeResult]


@dataclass(frozen=True)
class WorkflowEvent:
    sequence: int
    run_id: str
    node_id: str
    kind: str
    status: NodeStatus
    timestamp: float
    detail: Any = None


class ActionCancelled(Exception):
    """Adapter confirms the operation is quiescent after cooperative cancellation."""


class ExecutionUncertain(Exception):
    """An operation may still be active: keep its resources quarantined."""


class ActionOutputError(ValueError):
    """A completed action returned data that violates its declared output schema."""


class PayloadLimitError(ValueError):
    """A JSON input, output or feedback report exceeds its configured byte budget."""
