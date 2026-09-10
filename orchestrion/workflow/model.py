"""Transport-independent workflow definitions and execution snapshots."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple, Union


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a nonempty string".format(label))


def _positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer".format(label))


def _timeout(value: Optional[float]) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        or value > 1e12
        or not math.isfinite(value)
    ):
        raise ValueError("timeout must be finite, positive, at most 1e12, or None")


@dataclass(frozen=True)
class Resource:
    """A named capacity shared by every workflow in one engine."""

    name: str
    capacity: int = 1

    def __post_init__(self) -> None:
        _name(self.name, "resource name")
        _positive_int(self.capacity, "resource capacity")


@dataclass(frozen=True)
class ResourceClaim:
    resource: str
    units: int = 1

    def __post_init__(self) -> None:
        _name(self.resource, "claimed resource")
        _positive_int(self.units, "claim units")


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


def _snapshot(value: Any, *, references: bool = False) -> Any:
    """Copy finite JSON without deepcopy hooks; reject cycles and nonfinite data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    ancestors = set()

    def visit(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float) and math.isfinite(item):
            return item
        if references and isinstance(item, OutputRef):
            return item
        if isinstance(item, (dict, list, tuple)):
            if id(item) in ancestors:
                raise ValueError("workflow data must not contain cycles")
            ancestors.add(id(item))
            try:
                if isinstance(item, dict):
                    if any(not isinstance(key, str) for key in item):
                        raise TypeError("workflow data requires string dictionary keys")
                    return {key: visit(child) for key, child in item.items()}
                return [visit(child) for child in item]
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
        _timeout(self.timeout)


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
        _timeout(self.timeout)


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
class NodeResult:
    node_id: str
    action: str
    status: NodeStatus = NodeStatus.PENDING
    output: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    feedback_dropped: int = 0


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
