"""Validate the whole graph before any adapter can perform device I/O."""

from collections import deque
from dataclasses import dataclass, replace
from hashlib import sha256
from json import dumps
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from orchestrion.schema import CommandValidationError

from .actions import Action
from .model import Node, OutputRef, Resource, ResourceClaim, Workflow, _snapshot


@dataclass(frozen=True)
class CompiledNode:
    node: Node
    dependencies: Tuple[str, ...]
    claims: Tuple[ResourceClaim, ...]
    action_version: Optional[str] = None


def _is_output_reference(value: Any) -> bool:
    return isinstance(value, OutputRef)


def _output_references(value: Any):
    if isinstance(value, OutputRef):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _output_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _output_references(child)


def merge_claims(*groups: Tuple[ResourceClaim, ...]) -> Tuple[ResourceClaim, ...]:
    """An action's mandatory claim is a floor; node claims can only increase it."""
    claims: Dict[str, int] = {}
    for group in groups:
        seen = set()
        for claim in group:
            if claim.resource in seen:
                raise ValueError("duplicate resource claim: " + claim.resource)
            seen.add(claim.resource)
            claims[claim.resource] = max(claim.units, claims.get(claim.resource, 0))
    return tuple(ResourceClaim(name, units) for name, units in sorted(claims.items()))


def check_claims(
    claims: Tuple[ResourceClaim, ...], resources: Mapping[str, Resource]
) -> None:
    for claim in claims:
        if claim.resource not in resources:
            raise ValueError("unknown resource: " + claim.resource)
        if claim.units > resources[claim.resource].capacity:
            raise ValueError("claim exceeds capacity: " + claim.resource)


def compile_workflow(
    workflow: Workflow,
    actions: Mapping[str, Action],
    resources: Mapping[str, Resource],
    max_nodes: int,
    max_input_bytes: Optional[int] = None,
) -> Tuple[CompiledNode, ...]:
    """Compile against the validated action registry owned by a WorkflowEngine."""
    if not isinstance(workflow, Workflow):
        raise TypeError("expected a Workflow")
    if len(workflow.nodes) > max_nodes:
        raise ValueError("workflow exceeds max_nodes_per_run")
    nodes: Dict[str, CompiledNode] = {}
    references_by_node = {}
    for original in workflow.nodes:
        # Re-snapshot even frozen definitions: nested dictionaries are mutable.
        node = replace(original)
        if max_input_bytes is not None:
            _snapshot(node.inputs, references=True, max_bytes=max_input_bytes)
        if node.node_id in nodes:
            raise ValueError("duplicate node ID: " + node.node_id)
        if node.action not in actions:
            raise ValueError("unknown action: " + node.action)
        action = actions[node.action]
        references = tuple(_output_references(node.inputs))
        # Registered action claims are immutable and already normalized. Only
        # node-level overrides require merging and allocating new claim objects.
        claims = (
            merge_claims(action.claims, node.claims) if node.claims else action.claims
        )
        check_claims(claims, resources)
        schema = action.input_schema
        if schema is not None:
            if references:
                schema.validate_template(node.inputs, is_deferred=_is_output_reference)
            else:
                schema.validate(dict(node.inputs))
            # Schema overrides may replace nested references during validation.
            references = tuple(_output_references(node.inputs))
        if references:
            references_by_node[node.node_id] = references
        dependencies = tuple(
            dict.fromkeys(
                (*node.after, *(reference.node_id for reference in references))
            )
        )
        nodes[node.node_id] = CompiledNode(node, dependencies, claims, action.version)

    successors = {node_id: [] for node_id in nodes}
    indegree = {}
    for node_id, compiled in nodes.items():
        indegree[node_id] = len(compiled.dependencies)
        for dependency in compiled.dependencies:
            if dependency not in nodes:
                raise ValueError("unknown dependency: " + dependency)
            successors[dependency].append(node_id)
        # Reuse discovery after all nodes exist, including forward references.
        for reference in references_by_node.get(node_id, ()):
            schema = actions[nodes[reference.node_id].node.action].output_schema
            if schema is not None:
                try:
                    schema.validate_path(reference.path)
                except CommandValidationError as exc:
                    raise ValueError(
                        "invalid output reference from {!r} to {!r}: {}".format(
                            reference.node_id, node_id, exc
                        )
                    ) from exc

    ready = deque(node_id for node_id in nodes if indegree[node_id] == 0)
    ordered = []
    while ready:
        node_id = ready.popleft()
        ordered.append(nodes[node_id])
        for successor in successors[node_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(ordered) != len(nodes):
        raise ValueError("workflow dependencies contain a cycle")
    return tuple(ordered)


def resolve_inputs(value: Any, get_output: Callable[[str], Any]) -> Any:
    """Copy inputs, looking up upstream output only where a reference needs it."""
    if isinstance(value, OutputRef):
        result = get_output(value.node_id)
        for index, part in enumerate(value.path):
            try:
                if isinstance(part, str) and isinstance(result, dict):
                    result = result[part]
                elif isinstance(part, int) and isinstance(result, list):
                    result = result[part]
                else:
                    raise ValueError("output path type mismatch")
            except (KeyError, IndexError, ValueError) as exc:
                # Keep exception categories compatible while identifying the
                # upstream node and the first path component that failed.
                raise type(exc)(
                    "cannot resolve output from node {!r} at path {!r}: {}".format(
                        value.node_id, value.path[: index + 1], exc
                    )
                ) from exc
        return _snapshot(result)
    if isinstance(value, dict):
        return {key: resolve_inputs(child, get_output) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_inputs(child, get_output) for child in value]
    return value


def describe_inputs(value: Any) -> Any:
    """Render input bindings in a JSON preview; this is not an executable codec."""
    if isinstance(value, OutputRef):
        return {"$output": {"node_id": value.node_id, "path": list(value.path)}}
    if isinstance(value, dict):
        return {key: describe_inputs(child) for key, child in value.items()}
    if isinstance(value, list):
        return [describe_inputs(child) for child in value]
    return value


def submission_fingerprint(
    workflow: Workflow, compiled: Tuple[CompiledNode, ...]
) -> str:
    """Identify the execution snapshot without retaining potentially large inputs.

    Type tags distinguish output references from literal preview-shaped JSON.
    Node order matters because it determines FIFO admission among equal priorities.
    """

    def encode(value):
        if isinstance(value, OutputRef):
            return ["reference", value.node_id, list(value.path)]
        if isinstance(value, dict):
            return ["object", [[key, encode(value[key])] for key in sorted(value)]]
        if isinstance(value, list):
            return ["array", [encode(child) for child in value]]
        return ["value", value]

    definition = [
        workflow.failure_policy.value,
        workflow.timeout,
        [
            [
                item.node.node_id,
                item.node.action,
                item.action_version,
                encode(item.node.inputs),
                sorted(item.dependencies),
                [(claim.resource, claim.units) for claim in item.claims],
                item.node.priority,
                item.node.timeout,
                item.node.stop_timeout,
            ]
            for item in compiled
        ],
    ]
    encoded = dumps(
        definition, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    return sha256(encoded.encode("ascii")).hexdigest()
